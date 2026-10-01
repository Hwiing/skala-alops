"""
POST /predict - 최근 120일(날짜 포함)로 다음 1~4주 경유 평균가 예측 (contracts.md v2)
POST /predict/batch-test - 드리프트 감지 시뮬레이션 시작점 (scripts/simulate_drift.py 참고)
"""

from datetime import date, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from data.diesel_features import INPUT_DAYS, WINDOWS
from serving_app import model_loader
from serving_app.monitoring import prediction_window, retrain_trigger
from serving_app.monitoring.prediction_window import PredictionWindow
from serving_app.schemas import (
    BatchPair,
    BatchTestRequest,
    BatchTestResponse,
    DriftCheck,
    PredictionValues,
    PredictRequest,
    PredictResponse,
    WeekPrediction,
)

router = APIRouter()

# 최근 예측 기록. AIOps 내부 윈도우(#15)에 모델 버전·출처(batch)와 함께 남긴다.
# 같은 기준일은 덮어쓰고(재집계 방지), 잠금으로 동시 배치를 보호한다. HTTP 응답의 BatchPair와는 별개 타입.
recent_predictions: PredictionWindow = prediction_window.window


def _get_model_or_503():
    try:
        return model_loader.get_model()
    except (FileNotFoundError, NotImplementedError, ImportError) as exc:
        raise HTTPException(
            503, "모델이 준비되지 않았습니다. 학습 또는 MLflow TODO를 완료하세요."
        ) from exc


def _weeks(base: date, values: list[float]) -> list[WeekPrediction]:
    """k주 = base+7(k−1)+1 ~ base+7k."""
    return [
        WeekPrediction(
            horizon_week=k,
            start_date=base + timedelta(days=w[0]),
            end_date=base + timedelta(days=w[-1]),
            predicted_avg_price=round(v, 2),
        )
        for k, (w, v) in enumerate(zip(WINDOWS, values), start=1)
    ]


def _predict_values(model, rows: list[dict]) -> list[float]:
    try:
        return PredictionValues(values=model.predict(rows)).values
    except (ValidationError, ValueError, TypeError) as exc:
        raise HTTPException(503, "모델 출력이 경유 4주 예측 계약에 맞지 않습니다") from exc


def _pair_with_actual(model, rows: list[dict]) -> list[dict]:
    """rows[i:i+120]마다 1~4주를 예측하고, 기준일 뒤 k주 실제 평균·naive(기준일 가격)를 붙인다."""
    prices = [r["diesel_price"] for r in rows]
    pairs = []
    for b in range(INPUT_DAYS - 1, len(rows) - 7 * len(WINDOWS)):
        pairs.append(
            {
                "date": rows[b]["date"],
                "predicted": [
                    round(v, 2) for v in _predict_values(model, rows[b - INPUT_DAYS + 1 : b + 1])
                ],
                "actual": [round(sum(prices[b + d] for d in w) / 7, 2) for w in WINDOWS],
                "naive": prices[b],
            }
        )
    return [BatchPair.model_validate(p).model_dump(mode="json") for p in pairs]


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = _get_model_or_503()
    base = req.sequence[-1].date
    values = _predict_values(model, [p.model_dump(mode="json") for p in req.sequence])
    # 실시간 예측은 정답 없이 기록하고, 나중에 /data/upload로 정답이 들어오면 채운다.
    recent_predictions.record(
        base.isoformat(),
        [round(v, 2) for v in values],
        req.sequence[-1].diesel_price,
        model.version,
    )
    return PredictResponse(
        predictions=_weeks(base, values), base_date=base, model_version=model.version
    )


@router.post(
    "/predict/batch-test", response_model=BatchTestResponse, response_model_exclude_none=True
)
def batch_test(req: BatchTestRequest):
    """기준일을 하루씩 밀며 1~4주를 예측하고 AIOps 판정(check_and_trigger)을 돌려준다.

    재학습으로 새 버전이 승격되면 그 버전을 서빙하도록 교체하고(reload_model),
    실제로 그 버전이 올라왔을 때만 이전 모델의 예측 기록을 비운다(#14 역할 분담).
    """
    model = _get_model_or_503()
    pairs = _pair_with_actual(model, [p.model_dump(mode="json") for p in req.rows])
    recent_predictions.record_batch(pairs, model.version)
    return BatchTestResponse(predictions=pairs, drift_check=judge_and_swap())


def fill_live_actuals(rows: list[dict]) -> int:
    """업로드된 일별 가격으로 실시간 예측의 k주 정답을 채운다. 7일이 모두 있는 주만. 채운 주 수를 돌려준다."""
    prices = {r["date"]: float(r["diesel_price"]) for r in rows}
    filled = 0
    for p in recent_predictions.pairs():
        if p["source"] != "live":
            continue
        base = date.fromisoformat(p["date"])
        for k, w in enumerate(WINDOWS):
            days = [(base + timedelta(days=d)).isoformat() for d in w]
            if p["actual"][k] is None and all(d in prices for d in days):
                avg = round(sum(prices[d] for d in days) / len(days), 2)
                filled += recent_predictions.fill_actual(p["date"], k + 1, avg)
    return filled


def judge_and_swap() -> dict:
    """기록 전체로 AIOps 판정을 돌리고, 승격되면 그 버전으로 교체한다. 교체 성공 때만 기록을 비운다."""
    try:
        drift_check = DriftCheck.model_validate(
            retrain_trigger.check_and_trigger(recent_predictions.pairs())
        ).model_dump(exclude_none=True)
    except NotImplementedError as exc:
        raise HTTPException(501, f"AIOps 판정 미구현: {exc}") from exc
    except ValidationError as exc:
        raise HTTPException(503, "AIOps 결과가 공통 계약에 맞지 않습니다") from exc

    if drift_check.get("promoted"):
        drift_check["reload"] = model_loader.reload_model(drift_check.get("version"))
        try:
            DriftCheck.model_validate(drift_check)
        except ValidationError as exc:
            raise HTTPException(503, "모델 교체 결과가 공통 계약에 맞지 않습니다") from exc
        retrain_trigger.notify_reload(drift_check, recent_predictions.pairs())
        if drift_check["reload"]["reloaded"]:
            recent_predictions.clear()
    return drift_check
