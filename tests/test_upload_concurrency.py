"""실제 이벤트 루프에서 업로드의 느린 재학습 중에도 health 요청이 진행되는지 확인한다."""

import asyncio
import threading

import httpx

from serving_app.main import app
from serving_app.routers import data as data_router
from serving_app.routers import predict as predict_router
from tests.test_scaffold import _csv, make_rows


def test_health_responds_while_upload_retraining_waits(monkeypatch, tmp_path):
    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(data_router, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(predict_router, "fill_live_actuals", lambda rows: 1)

    def slow_judge():
        started.set()
        if not release.wait(3):
            raise RuntimeError("health did not respond during retraining")
        return {"status": "ok"}

    monkeypatch.setattr(predict_router, "judge_and_swap", slow_judge)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            upload = asyncio.create_task(
                client.post("/data/upload", files={"file": ("rows.csv", _csv(make_rows(175)))})
            )
            try:
                assert await asyncio.to_thread(started.wait, 2)
                health = await asyncio.wait_for(client.get("/health"), timeout=1)
                assert health.status_code == 200 and not upload.done()
            finally:
                release.set()
            assert (await upload).status_code == 200

    asyncio.run(exercise())
