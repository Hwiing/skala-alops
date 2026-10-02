"""대시보드 운영 현황 API: GET /models(레지스트리·재학습 이력), GET /metrics/summary(요청 지표).

CI는 mlflow 없이 돌아가므로 /models는 가짜 mlflow 모듈로 확인한다.
"""

import json
import sys
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from serving_app import model_loader
from serving_app.main import app
from serving_app.routers import metrics

client = TestClient(app)


def _run(run_id, start, mode, gate, rmse, naive, before="none", reasons=""):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=start),
        data=SimpleNamespace(
            params={
                "mode": mode,
                "production_before": before,
                "data_period": "2024-01-01 ~ 2026-02-28",
            },
            metrics={
                **{f"rmse_w{k}": v for k, v in enumerate(rmse, start=1)},
                **{f"naive_rmse_w{k}": v for k, v in enumerate(naive, start=1)},
            },
            tags={"gate": gate, "gate_reasons": reasons},
        ),
    )


@pytest.fixture
def registry(monkeypatch):
    """v1(scratch) → v2(fine-tune, champion) 등록, 그 사이 게이트 실패 run 하나."""
    runs = [
        _run("r3", 3_000, "fine-tune", "passed", [30, 50, 60, 70], [35, 60, 80, 90], before="1"),
        _run(
            "r2",
            2_000,
            "fine-tune",
            "failed",
            [60, 1, 1, 1],
            [35, 60, 80, 90],
            before="1",
            reasons="1주차 RMSE 60.00 > 50",
        ),
        _run("r1", 1_000, "scratch", "passed", [20, 40, 50, 60], [25, 50, 70, 80]),
    ]
    versions = [
        SimpleNamespace(
            version=2,
            run_id="r3",
            current_stage="Production",
            aliases=[],  # 실제 MLflow 검색 결과는 alias를 채우지 않는다
            creation_timestamp=3_500,
        ),
        SimpleNamespace(
            version=1, run_id="r1", current_stage="Archived", aliases=[], creation_timestamp=1_500
        ),
    ]
    calls = {}

    class FakeClient:
        def search_experiments(self):
            return [SimpleNamespace(experiment_id="0")]

        def search_runs(self, experiment_ids, filter_string, order_by, max_results):
            calls["runs"] = (experiment_ids, filter_string)
            return runs

        def search_model_versions(self, filter_string):
            calls["versions"] = filter_string
            return versions

        def get_registered_model(self, name):
            return SimpleNamespace(aliases={"champion": "2"})

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(set_tracking_uri=lambda uri: None))
    monkeypatch.setitem(sys.modules, "mlflow.tracking", SimpleNamespace(MlflowClient=FakeClient))
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.setattr(model_loader, "_model_cache", SimpleNamespace(version="champion:2"))
    return calls


def test_models_lists_runs_newest_first_with_registry_fields(registry):
    body = client.get("/models").json()

    assert body["source"] == "mlflow"
    assert body["serving_version"] == "champion:2"
    assert body["production_version"] == "2"
    assert [h["run_id"] for h in body["history"]] == ["r3", "r2", "r1"]
    latest, failed, base = body["history"]
    assert latest["version"] == "2" and latest["stage"] == "Production"
    assert latest["aliases"] == ["champion"]
    assert latest["mode"] == "fine-tune" and latest["base_version"] == "1"
    assert latest["rmse"] == [30, 50, 60, 70] and latest["naive_rmse"] == [35, 60, 80, 90]
    assert latest["gate"] == "passed"
    assert latest["started_at"] and latest["registered_at"]
    # 게이트 실패 run은 등록되지 않아 버전이 없다
    assert failed["gate"] == "failed" and failed["version"] is None
    assert failed["gate_reasons"] == "1주차 RMSE 60.00 > 50"
    assert base["mode"] == "scratch" and base["base_version"] is None
    assert registry["versions"] == "name='DieselPricePredictor'"


def test_models_in_local_mode_has_no_registry(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setitem(sys.modules, "mlflow", None)  # local 모드는 mlflow를 부르지 않아야 한다

    body = client.get("/models").json()

    assert body["source"] == "local" and body["history"] == []
    assert body["production_version"] is None


def test_models_reports_registry_error_without_500(registry, monkeypatch):
    def boom(self):
        raise ConnectionError("db locked")

    monkeypatch.setattr(sys.modules["mlflow.tracking"].MlflowClient, "search_experiments", boom)

    res = client.get("/models")

    assert res.status_code == 200
    assert res.json()["history"] == [] and "db locked" in res.json()["error"]


# ---------------- /metrics/summary ----------------


@pytest.fixture
def request_log(tmp_path, monkeypatch):
    path = tmp_path / "requests.log"
    monkeypatch.setattr(metrics, "REQUESTS_LOG", str(path))
    return path


def test_middleware_records_tracked_paths_only(request_log):
    client.post("/predict", json={})  # 422도 기록 대상
    client.get("/health")
    client.get("/metrics/summary")

    lines = [json.loads(line) for line in request_log.read_text().splitlines()]
    assert [(r["path"], r["status"]) for r in lines] == [("/predict", 422)]
    assert lines[0]["duration_ms"] >= 0


def test_summary_aggregates_per_path_within_window(request_log):
    now = time.time()
    rows = [
        {"ts": now - 10, "path": "/predict", "status": 200, "duration_ms": d}
        for d in (10, 20, 30, 40, 100)
    ] + [
        {"ts": now - 10, "path": "/predict", "status": 503, "duration_ms": 5},
        {"ts": now - 7200, "path": "/predict", "status": 500, "duration_ms": 9999},  # 5m 밖
        {"ts": now - 10, "path": "/predict/batch-test", "status": 200, "duration_ms": 30000},
    ]
    request_log.write_text("".join(json.dumps(r) + "\n" for r in rows))

    body = client.get("/metrics/summary", params={"window": "5m"}).json()

    p = body["paths"]["/predict"]
    assert p["requests"] == 6
    assert p["success_rate"] == pytest.approx(5 / 6)
    assert p["p50_ms"] == 20 and p["p95_ms"] == 100
    assert body["paths"]["/predict/batch-test"]["p95_ms"] == 30000  # 실시간 예측과 섞지 않는다
    assert body["paths"]["/data/upload"] == {
        "requests": 0,
        "success_rate": None,
        "p50_ms": None,
        "p95_ms": None,
    }


def test_summary_without_log_file_and_bad_window(request_log):
    body = client.get("/metrics/summary", params={"window": "1h"}).json()
    assert body["paths"]["/predict"]["requests"] == 0

    assert client.get("/metrics/summary", params={"window": "7d"}).status_code == 422


def test_tests_do_not_write_runtime_logs():
    """테스트가 운영 로그(logs/aiops.log·requests.log)를 오염시키지 않는다 (conftest)."""
    import logging
    import os

    assert metrics.REQUESTS_LOG != os.path.join("logs", "requests.log")
    files = [
        h.baseFilename for h in logging.getLogger("aiops").handlers if hasattr(h, "baseFilename")
    ]
    assert not any(os.path.abspath("logs") in f for f in files)
    assert not os.getenv("AIOPS_ALERT_FILE", "logs/alerts.jsonl").startswith("logs")
