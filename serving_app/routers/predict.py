"""
POST /predict - 최근 120일(날짜 포함)로 다음 1~4주 경유 평균가 예측 (contracts.md v2)
POST /predict/batch-test - 드리프트 감지 시뮬레이션 시작점 (scripts/simulate_drift.py 참고)
"""

from datetime import date, timedelta

from fastapi import APIRouter, HTTPException

from data.diesel_features import INPUT_DAYS, WINDOWS
from serving_app import model_loader
from serving_app.monitoring import retrain_trigger
from serving_app.schemas import (
    PAIR_WINDOW,
    BatchTestRequest,
    BatchTestResponse,
    PredictRequest,
    PredictResponse,
    WeekPrediction,
)

router = APIRouter()

# 최근 예측 짝({date, predicted[4], actual[4], naive})을 PAIR_WINDOW(28)건만 유지한다.
# ponytail: 전역 리스트·잠금 없음(단일 사용자 시연용), 동시 배치 요청이 생기면 잠금 추가.
recent_predictions: list[dict] = []


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


def _pair_with_actual(model, rows: list[dict]) -> list[dict]:
    """rows[i:i+120]마다 1~4주를 예측하고, 기준일 뒤 k주 실제 평균·naive(기준일 가격)를 붙인다."""
    prices = [r["diesel_price"] for r in rows]
    pairs = []
    for b in range(INPUT_DAYS - 1, len(rows) - 7 * len(WINDOWS)):
        pairs.append(
            {
                "date": rows[b]["date"],
                "predicted": [round(v, 2) for v in model.predict(rows[b - INPUT_DAYS + 1 : b + 1])],
                "actual": [round(sum(prices[b + d] for d in w) / 7, 2) for w in WINDOWS],
                "naive": prices[b],
            }
        )
    return pairs


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = _get_model_or_503()
    base = req.sequence[-1].date
    values = model.predict([p.model_dump(mode="json") for p in req.sequence])
    return PredictResponse(
        predictions=_weeks(base, values), base_date=base, model_version=model.version
    )


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """기준일을 하루씩 밀며 1~4주를 예측하고 AIOps 판정(check_and_trigger)을 돌려준다.

    재학습으로 새 버전이 승격되면 그 버전을 서빙하도록 교체하고(reload_model),
    실제로 그 버전이 올라왔을 때만 이전 모델의 예측 기록을 비운다(#14 역할 분담).
    """
    model = _get_model_or_503()
    pairs = _pair_with_actual(model, [p.model_dump(mode="json") for p in req.rows])
    recent_predictions.extend(pairs)
    del recent_predictions[:-PAIR_WINDOW]

    try:
        drift_check = retrain_trigger.check_and_trigger(recent_predictions)
    except NotImplementedError as exc:
        raise HTTPException(501, f"AIOps 판정 미구현: {exc}") from exc

    if drift_check.get("promoted"):
        drift_check["reload"] = model_loader.reload_model(drift_check.get("version"))
        if drift_check["reload"]["reloaded"]:
            recent_predictions.clear()

    return BatchTestResponse(predictions=pairs, drift_check=drift_check)
