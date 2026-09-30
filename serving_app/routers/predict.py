"""
Day1 -> Day3(시뮬레이션 엔드포인트 추가) 확장 파일.

Day1: POST /predict - 최근 SEQ_LEN(20)일 시퀀스로 다음날 휘발유 가격 예측
Day3: POST /predict/batch-test - 드리프트 감지 시뮬레이션 시작점 (scripts/simulate_drift.py 참고)
"""

from fastapi import APIRouter, HTTPException

from serving_app import model_loader
from serving_app.schemas import BatchTestRequest, BatchTestResponse, PredictRequest, PredictResponse

router = APIRouter()

# Day3: 최근 예측 기록(actual/predicted)을 쌓아두는 슬라이딩 윈도우.
# monitoring/drift_detector.py의 WINDOW_SIZE(21)만큼만 유지한다.
recent_predictions: list[dict] = []


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = model_loader.get_model()
    except (FileNotFoundError, NotImplementedError, ImportError) as exc:
        raise HTTPException(
            503, "모델이 준비되지 않았습니다. 학습 또는 MLflow TODO를 완료하세요."
        ) from exc
    sequence = [p.model_dump() for p in req.sequence]
    predicted_price = model.predict_one(sequence)
    return PredictResponse(predicted_price=round(predicted_price, 2), model_version=model.version)


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """TODO(hootbee + kchanis1223): req.rows의 20일 윈도우를 순서대로 예측.

    다음 행 gasoline_price를 actual로 연결하고 recent_predictions 최근 21건 유지.
    check_and_trigger 결과를 반환. 원본 TODO 미구현 상태를 성공으로 표시하지 않는다.
    """
    raise HTTPException(501, "TODO: batch_test 슬라이딩 예측 및 드리프트 연결")
