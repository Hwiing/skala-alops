import logging
import os
import tempfile

import pytest

# 운영자 수신함(JsonlAdapter)은 retrain_trigger import 때 경로가 정해지므로 import 전에 임시 경로로 돌린다.
os.environ["AIOPS_ALERT_FILE"] = os.path.join(tempfile.mkdtemp(), "alerts.jsonl")

from serving_app.routers import metrics  # noqa: E402
from serving_app.routers import predict as predict_router  # noqa: E402


@pytest.fixture(autouse=True)
def _no_pending_judgement(monkeypatch):
    """판정 미완료 상태는 프로세스 전역이라 테스트 사이에 새지 않게 매번 초기화한다."""
    monkeypatch.setattr(predict_router, "_judgement_pending", False)


@pytest.fixture(autouse=True)
def _no_runtime_logs(tmp_path_factory, monkeypatch):
    """테스트가 운영 로그(logs/aiops.log·requests.log)에 쓰지 않게 한다. 대시보드가 그 파일을 그대로 보여준다."""
    monkeypatch.setattr(
        metrics, "REQUESTS_LOG", str(tmp_path_factory.mktemp("logs") / "requests.log")
    )
    logger = logging.getLogger("aiops")
    for handler in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
        logger.removeHandler(handler)  # serving_app.main import 때 붙은 logs/aiops.log 핸들러
        handler.close()
