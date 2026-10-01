"""
Day1 -> Day3(시뮬레이션 엔드포인트 추가) 확장 파일.

Day1: POST /predict - 최근 SEQ_LEN(20)일 시퀀스로 다음날 휘발유 가격 예측
Day3: POST /predict/batch-test - 드리프트 감지 시뮬레이션 시작점 (scripts/simulate_drift.py 참고)
"""

from fastapi import APIRouter, HTTPException

from data.features import FEATURE_COLUMNS, SEQ_LEN
from serving_app import model_loader
from serving_app.monitoring import retrain_trigger
from serving_app.monitoring.drift_detector import WINDOW_SIZE
from serving_app.schemas import BatchTestRequest, BatchTestResponse, PredictRequest, PredictResponse

router = APIRouter()

# Day3: 최근 예측 기록(actual/predicted)을 쌓아두는 슬라이딩 윈도우.
# monitoring/drift_detector.py의 WINDOW_SIZE(21)만큼만 유지한다.
# ponytail: 전역 리스트·잠금 없음(단일 사용자 시연용), 동시 배치 요청이 생기면 잠금 추가.
recent_predictions: list[dict] = []


def _get_model_or_503():
    try:
        return model_loader.get_model()
    except (FileNotFoundError, NotImplementedError, ImportError) as exc:
        raise HTTPException(
            503, "모델이 준비되지 않았습니다. 학습 또는 MLflow TODO를 완료하세요."
        ) from exc


def _pair_with_actual(model, rows: list[dict]) -> list[dict]:
    """SEQ_LEN일 윈도우마다 다음날 가격을 예측하고 바로 다음 행 가격을 actual로 붙인다.

    #28 주간 예측 전환 시 이 함수만 바꾼다(actual = 해당 주차 실제 주간평균).
    """
    target = FEATURE_COLUMNS[0]
    return [
        {"predicted": model.predict_one(rows[i : i + SEQ_LEN]), "actual": rows[i + SEQ_LEN][target]}
        for i in range(len(rows) - SEQ_LEN)
    ]


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = _get_model_or_503()
    sequence = [p.model_dump() for p in req.sequence]
    predicted_price = model.predict_one(sequence)
    return PredictResponse(predicted_price=round(predicted_price, 2), model_version=model.version)


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """rows를 20일 윈도우로 밀며 예측하고 AIOps 판정(check_and_trigger)을 돌려준다.

    재학습으로 새 버전이 승격되면 서버 모델을 교체하고(reload_model),
    교체에 성공했을 때만 이전 모델의 예측 기록을 비운다(#14 역할 분담).
    """
    model = _get_model_or_503()
    pairs = _pair_with_actual(model, [p.model_dump() for p in req.rows])
    recent_predictions.extend(pairs)
    del recent_predictions[:-WINDOW_SIZE]

    try:
        drift_check = retrain_trigger.check_and_trigger(recent_predictions)
    except NotImplementedError as exc:
        raise HTTPException(501, f"AIOps 판정 미구현: {exc}") from exc

    if drift_check.get("promoted"):
        drift_check["reload"] = model_loader.reload_model()
        if drift_check["reload"]["reloaded"]:
            recent_predictions.clear()

    return BatchTestResponse(
        predictions=[round(p["predicted"], 2) for p in pairs], drift_check=drift_check
    )
