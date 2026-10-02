"""초기 학습·배포의 비동기 실행·실제 서빙 교체·실패 복구 회귀 검사."""

import csv
import io
import sys
import threading
import time
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from serving_app import model_loader
from serving_app.main import app
from serving_app.monitoring import retrain_trigger
from serving_app.monitoring.prediction_window import PredictionWindow
from serving_app.routers import data, predict, training
from serving_app.schemas import TrainingJob


def rows(count):
    return [
        {
            "date": (date(2024, 1, 1) + timedelta(days=i)).isoformat(),
            "diesel_price": 1500 + i,
            "singapore_diesel_price": 90,
            "usd_krw": 1400,
            "tax_or_supply_feature": 0,
        }
        for i in range(count)
    ]


def save_csv(path, count):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[*rows(1)[0], "provenance"], quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows([{**r, "provenance": 'source, "quoted"'} for r in rows(count)])
    path.write_text(output.getvalue())
    return str(path)


@pytest.fixture
def client(monkeypatch, tmp_path):
    path = save_csv(tmp_path / "uploaded.csv", 627)
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.setenv("LOADING_MODE", "lazy")
    monkeypatch.setattr(training, "latest_upload", lambda: path)
    monkeypatch.setattr(data, "latest_upload", lambda: path)
    monkeypatch.setattr(training, "_job", TrainingJob())
    monkeypatch.setattr(retrain_trigger, "_retrain_lock", threading.Lock())
    monkeypatch.setattr(retrain_trigger, "reset_state", lambda: None)
    monkeypatch.setattr(predict, "recent_predictions", PredictionWindow())
    monkeypatch.setattr(predict, "_judgement_pending", True)
    monkeypatch.setattr(model_loader, "_model_cache", SimpleNamespace(version="champion:1"))
    predict.recent_predictions.record("2024-01-01", [1500] * 4, 1500, "champion:1")
    with TestClient(app) as client:
        yield client


def trained():
    return dict(
        rmse=[1] * 4,
        naive_rmse=[2] * 4,
        run_id="training-run",
        model_uri="runs:/training-run/model",
        production_rmse=[3] * 4,
        production_before="1",
        passed=True,
        promoted=True,
        version="2",
        reasons=[],
    )


def wait_job(client):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = client.get("/training/status").json()
        if job["state"] != "running":
            # The worker releases the shared lock immediately after publishing its result.
            assert retrain_trigger._retrain_lock.acquire(timeout=1)
            retrain_trigger._retrain_lock.release()
            return job
        time.sleep(0.005)
    pytest.fail("training did not finish")


def test_training_remains_responsive_and_rejects_concurrent_jobs(client, monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def train(path, holdout):
        started.set()
        assert release.wait(3)
        assert holdout == 365
        return trained()

    monkeypatch.setattr(training, "_train", train)

    def reload(version):
        assert version == "2"
        model_loader._model_cache = SimpleNamespace(version="champion:2")
        return {"reloaded": True, "version": "champion:2"}

    monkeypatch.setattr(model_loader, "reload_model", reload)
    response = client.post("/training/start", json={})
    try:
        assert response.status_code == 202
        assert response.json()["state"] == "running"
        assert started.wait(1)
        assert client.get("/health").json()["model_version"] == "champion:1"
        assert client.get("/training/status").json()["state"] == "running"
        assert client.post("/training/start", json={}).status_code == 409
    finally:
        release.set()
    job = wait_job(client)
    assert job["state"] == "completed"
    assert job["result"]["promoted"] and job["result"]["version"] == "2"
    assert job["result"]["reload"]["reloaded"]
    assert client.get("/health").json()["model_version"] == "champion:2"
    assert len(predict.recent_predictions) == 0
    assert not predict.judgement_pending()


def test_empty_registry_first_deployment_loads_champion_one(client, monkeypatch):
    monkeypatch.setattr(model_loader, "_model_cache", None)
    result = {**trained(), "version": "1", "production_before": None, "production_rmse": None}
    monkeypatch.setattr(training, "_train", lambda *_: result)

    def reload(version):
        assert version == "1"
        model_loader._model_cache = SimpleNamespace(version="champion:1")
        return {"reloaded": True, "version": "champion:1"}

    monkeypatch.setattr(model_loader, "reload_model", reload)
    client.post("/training/start", json={})
    job = wait_job(client)
    assert job["state"] == "completed"
    assert job["result"]["reload"]["version"] == "champion:1"
    assert client.get("/health").json()["model_version"] == "champion:1"


def test_gate_failure_preserves_serving_and_prediction_records(client, monkeypatch):
    result = {
        **trained(),
        "rmse": [100] * 4,
        "passed": False,
        "promoted": False,
        "version": None,
        "reasons": ["worse than naive"],
    }
    monkeypatch.setattr(training, "_train", lambda *_: result)
    monkeypatch.setattr(model_loader, "reload_model", lambda *_: pytest.fail("unexpected reload"))
    client.post("/training/start", json={})
    job = wait_job(client)
    assert job["state"] == "completed" and not job["result"]["promoted"]
    assert job["result"]["reload"] is None
    assert client.get("/health").json()["model_version"] == "champion:1"
    assert len(predict.recent_predictions) == 1 and predict.judgement_pending()
    assert client.post("/training/reload", json={}).status_code == 409


def test_reload_failure_retries_same_version_without_training(client, monkeypatch):
    calls = []
    monkeypatch.setattr(training, "_train", lambda *_: calls.append("train") or trained())
    monkeypatch.setattr(
        model_loader,
        "reload_model",
        lambda *_: {
            "reloaded": False,
            "version": "champion:1",
            "error": "temporary loading failure",
        },
    )
    client.post("/training/start", json={})
    job = wait_job(client)
    assert job["state"] == "completed" and job["result"]["promoted"]
    assert not job["result"]["reload"]["reloaded"]
    assert len(predict.recent_predictions) == 1 and predict.judgement_pending()

    def reload(version):
        assert version == "2"
        model_loader._model_cache = SimpleNamespace(version="champion:2")
        return {"reloaded": True, "version": "champion:2"}

    monkeypatch.setattr(model_loader, "reload_model", reload)
    retried = client.post("/training/reload", json={})
    assert retried.status_code == 200 and retried.json()["result"]["reload"]["reloaded"]
    assert calls == ["train"]
    assert client.get("/health").json()["model_version"] == "champion:2"
    assert len(predict.recent_predictions) == 0 and not predict.judgement_pending()


def test_mismatching_loaded_version_is_not_success(client, monkeypatch):
    monkeypatch.setattr(training, "_train", lambda *_: trained())
    monkeypatch.setattr(
        model_loader, "reload_model", lambda *_: {"reloaded": True, "version": "champion:3"}
    )
    client.post("/training/start", json={})
    assert wait_job(client)["state"] == "failed"
    assert len(predict.recent_predictions) == 1 and predict.judgement_pending()


def test_training_worker_calls_register_entrypoint(monkeypatch):
    seen = []

    def initial(path, **kwargs):
        seen.append((path, kwargs))
        return {k: v for k, v in trained().items() if k != "model_uri"}

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(set_tracking_uri=lambda uri: None))
    monkeypatch.setitem(
        sys.modules, "serving_app.diesel_registry", SimpleNamespace(train_and_register=initial)
    )
    monkeypatch.setenv("DIESEL_DATA_SYNTHETIC", "false")
    assert training._train("uploaded.csv", 365) == trained()
    assert seen == [("uploaded.csv", {"holdout_days": 365, "synthetic": False})]


def test_training_exception_releases_lock_and_allows_retry(client, monkeypatch):
    def fail(*_):
        raise RuntimeError("resource temporarily unavailable")

    monkeypatch.setattr(training, "_train", fail)
    assert client.post("/training/start", json={}).status_code == 202
    assert wait_job(client)["state"] == "failed"
    assert client.post("/training/start", json={}).status_code == 202
    assert wait_job(client)["state"] == "failed"


@pytest.mark.parametrize("body", [{"holdout_days": 29}, {"unexpected": True}])
def test_invalid_training_options_rejected(client, body):
    assert client.post("/training/start", json=body).status_code == 422


def test_training_requires_sufficient_uploaded_data(client, monkeypatch, tmp_path):
    monkeypatch.setattr(training, "latest_upload", lambda: save_csv(tmp_path / "short.csv", 175))
    assert client.post("/training/start", json={}).status_code == 400
    monkeypatch.setenv("MODEL_SOURCE", "local")
    assert client.post("/training/start", json={}).status_code == 409


def test_preview_normalizes_quoted_extra_columns(client):
    response = client.get("/data/preview")
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == 627
    assert body["preview"] == rows(627)[-175:]
    assert "provenance" not in body["preview"][0]


def test_evaluation_does_not_trigger_or_record_aiops(client, monkeypatch):
    class Model:
        version = "champion:1"

        def predict(self, sequence):
            return [sequence[-1]["diesel_price"]] * 4

    monkeypatch.setattr(model_loader, "_model_cache", Model())
    monkeypatch.setattr(predict, "judge_and_swap", lambda: pytest.fail("unexpected AIOps"))
    response = client.post("/predict/evaluate", json={"rows": rows(175)})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "champion:1" and len(body["predictions"]) == 28
    assert body["predictions"][0]["actual"] == [1623, 1630, 1637, 1644]
    assert len(predict.recent_predictions) == 1 and predict.judgement_pending()
