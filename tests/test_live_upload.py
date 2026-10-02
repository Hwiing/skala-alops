"""#14 실시간 /predict 기록 → /data/upload 지연 정답 → 판정·교체 (#54 병합 후 리뷰).

실제 retrain_trigger(가짜 fine_tune)와 HTTP 경로로 확인한다.
    - 업로드 중 재학습이 서버 이벤트 루프를 막지 않는다
    - 교체 실패·판정 오류 뒤 같은 CSV를 다시 올리면 재학습 없이 교체·판정을 다시 시도한다
    - 같은 날짜를 다시 예측해도 이미 채운 정답은 지워지지 않는다
"""

import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from data.contracts import BATCH_MIN_ROWS, INPUT_DAYS, PAIR_WINDOW
from serving_app import model_loader
from serving_app.main import app
from serving_app.monitoring import retrain_trigger as rt
from serving_app.monitoring.prediction_window import PredictionWindow
from serving_app.routers import data as data_router
from serving_app.routers import predict as predict_router
from tests.test_drift_detector import _ShiftModel
from tests.test_retrain_trigger import PROMOTED as PROMOTED_RESULT
from tests.test_retrain_trigger import env  # noqa: F401  가짜 fine_tune·알림 fixture
from tests.test_scaffold import _csv, make_rows


@pytest.fixture
def live(env, monkeypatch, tmp_path):  # noqa: F811
    """기준일 28개를 /predict로 실시간 예측하고, 그 4주 정답이 모두 들어 있는 175행을 업로드한다.

    모델은 마지막 가격 + 60원(naive 오차 4원보다 나쁨) → drift → 가짜 fine_tune이 v5 승격.
    """
    reloads = []
    reload_ok = {"value": True}

    def fake_reload(version):
        reloads.append(version)
        if reload_ok["value"]:
            return {"reloaded": True, "version": f"champion:{version}"}
        return {"reloaded": False, "version": "champion:3", "error": "load failed"}

    monkeypatch.setattr(predict_router, "recent_predictions", PredictionWindow())
    monkeypatch.setattr(model_loader, "_model_cache", _ShiftModel(60.0, "champion:3"))
    monkeypatch.setattr(model_loader, "reload_model", fake_reload)
    monkeypatch.setattr(data_router, "UPLOAD_DIR", str(tmp_path))
    rows = make_rows(BATCH_MIN_ROWS)

    with TestClient(app) as client:  # 요청들이 같은 이벤트 루프를 공유해야 블로킹이 드러난다

        def predict(b):
            return client.post("/predict", json={"sequence": rows[b - INPUT_DAYS + 1 : b + 1]})

        def predict_all():
            for b in range(INPUT_DAYS - 1, INPUT_DAYS - 1 + PAIR_WINDOW):
                assert predict(b).status_code == 200

        def upload():
            response = client.post("/data/upload", files={"file": ("rows.csv", _csv(rows))})
            assert response.status_code == 200
            return response.json()

        yield SimpleNamespace(
            client=client,
            predict=predict,
            predict_all=predict_all,
            upload=upload,
            window=predict_router.recent_predictions,
            reloads=reloads,
            reload_ok=reload_ok,
            env=env,
        )


def test_upload_retrain_does_not_block_other_requests(live, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow_fine_tune(rows):
        started.set()
        release.wait(10)
        return {**live.env["result"]}

    monkeypatch.setattr(rt, "_fine_tune", slow_fine_tune)
    live.predict_all()
    result = {}
    worker = threading.Thread(target=lambda: result.update(live.upload()))
    worker.start()
    try:
        assert started.wait(5), "업로드가 재학습에 도달하지 못함"
        t0 = time.perf_counter()
        assert live.client.get("/health").status_code == 200
        assert live.predict(INPUT_DAYS - 1).status_code == 200
        assert time.perf_counter() - t0 < 1.0  # 재학습이 끝나기를 기다리지 않는다
    finally:
        release.set()
        worker.join(10)
    assert result["drift_check"]["status"] == "promoted"


def test_reupload_retries_failed_reload_without_retraining(live):
    live.predict_all()
    live.reload_ok["value"] = False
    first = live.upload()
    assert first["filled"] == PAIR_WINDOW * 4
    assert first["drift_check"]["reload"]["reloaded"] is False
    assert len(live.window) == PAIR_WINDOW  # 교체 실패 → 기록 유지

    live.reload_ok["value"] = True
    second = live.upload()  # 같은 CSV: 새 정답은 없지만 미완료 교체를 다시 시도
    assert second["filled"] == 0
    assert second["drift_check"]["reload"] == {"reloaded": True, "version": "champion:5"}
    assert live.reloads == ["5", "5"]
    assert len(live.env["calls"]) == 1  # 재학습은 한 번만 (이전 승격 결과 재사용)
    assert len(live.window) == 0

    third = live.upload()  # 모두 처리됨 → 판정하지 않음
    assert "drift_check" not in third and live.reloads == ["5", "5"]


def test_retrain_failure_keeps_judgement_pending_until_retried(live, monkeypatch):
    live.predict_all()
    live.env["error"] = RuntimeError("mlflow down")
    assert live.upload()["drift_check"]["status"] == "retrain_failed"

    monkeypatch.setattr(rt, "RETRAIN_COOLDOWN_SECONDS", 0)
    live.env["error"] = None
    second = live.upload()  # 새 정답은 없지만 재학습이 끝나지 않았으므로 다시 판정
    assert second["filled"] == 0
    assert second["drift_check"]["status"] == "promoted"
    assert len(live.env["calls"]) == 2


def test_judgement_error_after_fill_is_reported_and_retried(live, monkeypatch):
    live.predict_all()
    real = rt.check_and_trigger
    monkeypatch.setattr(rt, "check_and_trigger", lambda pairs: {"status": "retrain_triggered"})

    first = live.upload()  # 파일 저장·정답 적재는 성공, 판정만 실패
    assert first["filled"] == PAIR_WINDOW * 4
    assert "drift_check" not in first
    assert first["drift_check_error"]["status_code"] == 503

    monkeypatch.setattr(rt, "check_and_trigger", real)
    second = live.upload()
    assert second["filled"] == 0
    assert second["drift_check"]["status"] == "promoted"


def test_repredict_same_date_keeps_filled_actuals(live, monkeypatch):
    monkeypatch.setattr(model_loader, "_model_cache", _ShiftModel(0.0, "champion:3"))
    live.predict_all()
    assert live.upload()["drift_check"]["status"] == "ok"  # 교체 없음 → 기록 유지
    base = INPUT_DAYS - 1
    before = live.window.pairs()[0]["actual"]
    assert None not in before

    live.predict(base)  # 같은 기준일을 다시 예측

    assert live.window.pairs()[0]["actual"] == before
    assert "drift_check" not in live.upload()  # 판정 완료 상태 유지


def test_record_with_actual_still_updates_actual():
    window = PredictionWindow()
    window.record("2026-07-01", [1.0] * 4, 1.0, "v1", actual=[1.0, 2.0, None, None])
    window.record("2026-07-01", [2.0] * 4, 1.0, "v1")  # 실시간 재예측: 정답 없음 → 기존 유지
    window.record("2026-07-01", [3.0] * 4, 1.0, "v1", actual=[5.0, None, 7.0, None], source="batch")

    [record] = window.pairs()
    assert record["predicted"] == [3.0] * 4
    assert record["actual"] == [5.0, 2.0, 7.0, None]
    assert record["source"] == "batch"


@pytest.mark.parametrize(
    "failure",
    [
        {"result": {"status": "no_production", "promoted": False, "reasons": ["Production 없음"]}},
        {
            "error": ValueError("insufficient_data: 627행 필요, 400행")
        },  # 드리프트 뒤 재학습 데이터 부족
    ],
    ids=["no_production", "retrain_data_insufficient"],
)
def test_temporary_retrain_failure_after_drift_is_retried_on_reupload(live, monkeypatch, failure):
    """#14 리뷰(#59 코멘트) P2: 드리프트는 감지했지만 재학습을 못 한 상태는 다음 업로드가 다시 판정한다."""
    monkeypatch.setattr(rt, "RETRAIN_COOLDOWN_SECONDS", 0)
    live.predict_all()
    live.env.update(failure)
    first = live.upload()
    assert first["drift_check"]["status"] in ("no_production", "insufficient_data")
    assert "drift_rmse" in first["drift_check"]

    live.env.update(result={**live.env["result"], **PROMOTED_RESULT}, error=None)
    second = live.upload()  # 같은 CSV: 새 정답은 없지만 재학습이 끝나지 않았으므로 다시 판정
    assert second["filled"] == 0
    assert second["drift_check"]["status"] == "promoted"
    assert len(live.window) == 0


def test_too_few_pairs_does_not_keep_judgement_pending(live):
    """판정할 짝 자체가 부족하면(드리프트 판정 전 insufficient_data) 새 정답이 올 때까지 기다린다."""
    for b in range(INPUT_DAYS - 1, INPUT_DAYS - 1 + PAIR_WINDOW - 1):  # 27개만 예측
        assert live.predict(b).status_code == 200
    first = live.upload()
    assert first["drift_check"]["status"] == "insufficient_data"
    assert "drift_rmse" not in first["drift_check"]

    assert "drift_check" not in live.upload()
