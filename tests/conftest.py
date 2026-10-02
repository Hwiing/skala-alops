import pytest

from serving_app.routers import predict as predict_router


@pytest.fixture(autouse=True)
def _no_pending_judgement(monkeypatch):
    """판정 미완료 상태는 프로세스 전역이라 테스트 사이에 새지 않게 매번 초기화한다."""
    monkeypatch.setattr(predict_router, "_judgement_pending", False)
