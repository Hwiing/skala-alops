"""초기 학습의 비동기 실행·기존 모델 보존·독립 성능 검증 회귀 검사."""

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
        model_uri="models:/logged-model",
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
    monkeypatch.setattr(
        model_loader, "reload_model", lambda *_: pytest.fail("initial training changed serving")
    )
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
    assert job["result"] == trained()
    assert client.get("/health").json()["model_version"] == "champion:1"
    assert len(predict.recent_predictions) == 1
    assert predict.judgement_pending()
    assert "/training/reload" not in client.get("/openapi.json").json()["paths"]


def test_initial_training_records_metrics_without_a_gate(client, monkeypatch):
    result = {**trained(), "rmse": [100] * 4}
    monkeypatch.setattr(training, "_train", lambda *_: result)
    monkeypatch.setattr(model_loader, "reload_model", lambda *_: pytest.fail("unexpected reload"))
    client.post("/training/start", json={})
    job = wait_job(client)
    assert job["state"] == "completed" and job["result"] == result
    assert "passed" not in job["result"] and "promoted" not in job["result"]
    assert client.get("/health").json()["model_version"] == "champion:1"
    assert len(predict.recent_predictions) == 1


def test_initial_training_rejects_deployment_result(client, monkeypatch):
    monkeypatch.setattr(training, "_train", lambda *_: {**trained(), "promoted": True})
    client.post("/training/start", json={})
    assert wait_job(client)["state"] == "failed"
    assert client.get("/health").json()["model_version"] == "champion:1"
    assert len(predict.recent_predictions) == 1


def test_training_worker_calls_training_only_entrypoint(monkeypatch):
    seen = []

    def initial(path, **kwargs):
        seen.append((path, kwargs))
        return trained()

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(set_tracking_uri=lambda uri: None))
    monkeypatch.setitem(
        sys.modules, "serving_app.diesel_registry", SimpleNamespace(train_initial=initial)
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
