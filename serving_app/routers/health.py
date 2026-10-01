"""Day1: 헬스체크 엔드포인트.

status는 프로세스 생존(liveness)만 뜻한다. 모델 readiness는 model_loaded로 확인한다
(lazy 모드에서는 첫 /predict 전까지 false가 정상).
"""

import os

from fastapi import APIRouter

from serving_app import model_loader
from serving_app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    model = model_loader._model_cache
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": model.version if model is not None else None,
        "model_source": os.getenv("MODEL_SOURCE", "local"),
        "loading_mode": os.getenv("LOADING_MODE", "lazy"),
    }
