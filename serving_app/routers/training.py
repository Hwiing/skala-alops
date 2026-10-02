"""업로드 CSV의 초기 학습 작업. 학습·MLflow 기록만 수행하며 AIOps와 동시 학습을 막는다."""

import logging
import os
import threading
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from data.contracts import FINETUNE_MIN_ROWS, INPUT_DAYS, TARGET_DAYS
from data.diesel_features import load_diesel_rows
from data.storage import latest_upload
from serving_app.monitoring import retrain_trigger
from serving_app.schemas import TrainingJob, TrainingRequest, TrainingResult

router = APIRouter(prefix="/training")
logger = logging.getLogger("aiops")
_state_lock = threading.Lock()
_job = TrainingJob()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _train(path: str, holdout_days: int) -> dict:
    import mlflow

    from serving_app.diesel_registry import train_initial

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    return train_initial(
        path,
        holdout_days=holdout_days,
        synthetic=os.getenv("DIESEL_DATA_SYNTHETIC", "false").lower() == "true",
    )


def _run(path: str, holdout_days: int) -> None:
    global _job
    try:
        logger.info("[INFO] initial training started: %s", os.path.basename(path))
        result = TrainingResult.model_validate(_train(path, holdout_days))
        with _state_lock:
            _job = _job.model_copy(
                update={"state": "completed", "finished_at": _now(), "result": result}
            )
        logger.info("[INFO] initial training completed: run_id=%s", result.run_id)
    except Exception as exc:
        logger.exception("[ERROR] initial training failed")
        with _state_lock:
            _job = _job.model_copy(
                update={"state": "failed", "finished_at": _now(), "error": str(exc)}
            )
    finally:
        retrain_trigger._retrain_lock.release()


@router.get("/status", response_model=TrainingJob)
def status():
    with _state_lock:
        return _job.model_copy(deep=True)


@router.post("/start", status_code=202, response_model=TrainingJob)
def start(req: TrainingRequest):
    global _job
    if os.getenv("MODEL_SOURCE", "local") != "mlflow":
        raise HTTPException(409, "초기 학습은 모델 레지스트리 연동 모드에서 실행하세요.")
    try:
        path = latest_upload()
    except FileNotFoundError as exc:
        raise HTTPException(400, "CSV를 먼저 업로드하세요.") from exc
    rows = load_diesel_rows(path)
    minimum = max(FINETUNE_MIN_ROWS, INPUT_DAYS + req.holdout_days + 2 * TARGET_DAYS)
    if len(rows) < minimum:
        raise HTTPException(400, f"초기 학습에는 최소 {minimum}행이 필요합니다.")
    if not retrain_trigger._retrain_lock.acquire(blocking=False):
        raise HTTPException(409, "다른 학습이 진행 중입니다. 완료 후 다시 실행하세요.")
    with _state_lock:
        _job = TrainingJob(
            state="running",
            job_id=uuid4().hex,
            filename=os.path.basename(path),
            rows=len(rows),
            holdout_days=req.holdout_days,
            started_at=_now(),
        )
        response = _job.model_copy(deep=True)
    try:
        threading.Thread(target=_run, args=(path, req.holdout_days), daemon=True).start()
    except Exception:
        retrain_trigger._retrain_lock.release()
        with _state_lock:
            _job = _job.model_copy(
                update={"state": "failed", "error": "학습 작업을 시작하지 못했습니다."}
            )
        raise
    return response
