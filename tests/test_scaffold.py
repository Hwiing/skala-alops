import csv
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from data.features import FEATURE_COLUMNS, GasolineScaler, build_sequences, load_rows
from serving_app import model_loader
from serving_app.main import app
from serving_app.monitoring import retrain_trigger
from serving_app.routers import data as data_router
from serving_app.routers import predict as predict_router

SAMPLE = Path("data/sample_gasoline_prices.csv")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LOADING_MODE", "lazy")
    monkeypatch.setattr(model_loader, "_model_cache", None)
    with TestClient(app) as client:
        yield client


def payload(length=20):
    rows = load_rows(str(SAMPLE))
    return {"sequence": [{k: row[k] for k in FEATURE_COLUMNS} for row in rows[:length]]}


def test_dashboard_health_and_docs(client):
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200
    health = client.get("/health").json()
    assert health["model_loaded"] is False
    assert health["model_version"] is None
    assert health["model_source"] == "local"


@pytest.mark.parametrize("length", [19, 21])
def test_invalid_sequence_length(client, length):
    assert client.post("/predict", json=payload(length)).status_code == 422


@pytest.mark.parametrize("value", [0, -1, "NaN", "Infinity"])
def test_invalid_price(client, value):
    body = payload()
    body["sequence"][0]["gasoline_price"] = value
    assert client.post("/predict", json=body).status_code == 422


def test_predict_contract_with_injected_model(client, monkeypatch):
    class StubModel:
        version = "test-only"

        def predict_one(self, sequence):
            assert len(sequence) == 20
            assert set(sequence[0]) == set(FEATURE_COLUMNS)
            return 1712.4

    monkeypatch.setattr(model_loader, "_model_cache", StubModel())
    response = client.post("/predict", json=payload())
    assert response.status_code == 200
    assert response.json() == {"predicted_price": 1712.4, "model_version": "test-only"}
    assert client.get("/health").json()["model_version"] == "test-only"


def test_missing_model_returns_503(client, monkeypatch):
    def missing():
        raise FileNotFoundError("test missing model")

    monkeypatch.setattr(model_loader, "get_model", missing)
    assert client.post("/predict", json=payload()).status_code == 503


class SeqLastPriceModel:
    """윈도우 마지막 날 가격 + 1을 돌려주는 가짜 모델 (예측·실제값 짝을 확인하기 쉽게)."""

    version = "test-only"

    def predict_one(self, sequence):
        assert len(sequence) == 20
        return sequence[-1][FEATURE_COLUMNS[0]] + 1


@pytest.fixture
def batch(client, monkeypatch):
    """가짜 모델·트리거·reload로 batch_test를 돌리는 환경. state로 결과를 조작한다."""
    state = {"trigger": {"status": "ok"}, "reload": {"reloaded": True, "version": "production:2"}}
    calls = {"trigger": [], "reload": 0}

    def trigger(recent):
        calls["trigger"].append(list(recent))
        if state["trigger"] is NotImplementedError:
            raise NotImplementedError("TODO")
        return dict(state["trigger"])

    def reload():
        calls["reload"] += 1
        return dict(state["reload"])

    monkeypatch.setattr(model_loader, "_model_cache", SeqLastPriceModel())
    monkeypatch.setattr(retrain_trigger, "check_and_trigger", trigger)
    monkeypatch.setattr(model_loader, "reload_model", reload)
    monkeypatch.setattr(predict_router, "recent_predictions", [])

    def post(length=41):
        return client.post("/predict/batch-test", json={"rows": payload(length)["sequence"]})

    return SimpleNamespace(post=post, state=state, calls=calls)


def test_batch_pairs_each_prediction_with_next_row(batch):
    response = batch.post(41)

    assert response.status_code == 200
    prices = [row[FEATURE_COLUMNS[0]] for row in payload(41)["sequence"]]
    assert response.json()["predictions"] == [round(p + 1, 2) for p in prices[19:40]]
    assert predict_router.recent_predictions == [
        {"predicted": prices[i + 19] + 1, "actual": prices[i + 20]} for i in range(21)
    ]
    assert batch.calls["trigger"] == [predict_router.recent_predictions]


def test_batch_keeps_only_latest_window(batch):
    batch.post(30)
    batch.post(41)

    assert len(predict_router.recent_predictions) == 21


def test_batch_promoted_reloads_and_clears_window(batch):
    batch.state["trigger"] = {"status": "retrain_triggered", "promoted": True, "rmse": 8.0}

    response = batch.post(41)

    assert response.status_code == 200
    assert response.json()["drift_check"]["reload"] == {"reloaded": True, "version": "production:2"}
    assert batch.calls["reload"] == 1
    assert predict_router.recent_predictions == []


def test_batch_reload_failure_keeps_window(batch):
    batch.state["trigger"] = {"status": "retrain_triggered", "promoted": True, "rmse": 8.0}
    batch.state["reload"] = {"reloaded": False, "version": "production:1", "error": "boom"}

    response = batch.post(41)

    assert response.json()["drift_check"]["reload"]["reloaded"] is False
    assert len(predict_router.recent_predictions) == 21


def test_batch_not_promoted_skips_reload(batch):
    batch.state["trigger"] = {"status": "retrain_triggered", "promoted": False, "rmse": 12.0}

    response = batch.post(41)

    assert "reload" not in response.json()["drift_check"]
    assert batch.calls["reload"] == 0
    assert len(predict_router.recent_predictions) == 21


def test_batch_unimplemented_trigger_returns_501(batch):
    batch.state["trigger"] = NotImplementedError

    assert batch.post(41).status_code == 501


def test_batch_missing_model_returns_503(batch, monkeypatch):
    def missing():
        raise FileNotFoundError("test missing model")

    monkeypatch.setattr(model_loader, "get_model", missing)
    assert batch.post(41).status_code == 503


def test_csv_upload_and_status(client, monkeypatch, tmp_path):
    monkeypatch.setattr(data_router, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(data_router, "latest_upload", lambda: str(next(tmp_path.glob("*.csv"))))
    response = client.post("/data/upload", files={"file": ("sample.csv", SAMPLE.read_bytes())})
    assert response.status_code == 200
    assert response.json()["rows"] == 120
    status = client.get("/data/status").json()
    assert status["rows"] == 120
    assert status["min_price"] > 0


def test_invalid_csv_is_not_saved(client, monkeypatch, tmp_path):
    monkeypatch.setattr(data_router, "UPLOAD_DIR", str(tmp_path))
    rows = list(csv.DictReader(io.StringIO(SAMPLE.read_text())))
    rows[3]["date"] = rows[2]["date"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    response = client.post("/data/upload", files={"file": ("bad.csv", output.getvalue())})
    assert response.status_code == 400
    assert not list(tmp_path.iterdir())


def test_sequences_align_next_day_and_scaler_roundtrip(tmp_path):
    rows = load_rows(str(SAMPLE))
    scaler = GasolineScaler().fit(rows[:80])
    X, y = build_sequences(rows, scaler)
    assert len(X) == len(y) == 100
    assert len(X[0]) == 20 and len(X[0][0]) == 4
    assert y[0] == rows[20]["gasoline_price"]
    for row in (rows[0], rows[-1]):
        price = row["gasoline_price"]
        assert scaler.inverse_price(scaler.scale_price(price)) == pytest.approx(price)
    path = str(tmp_path / "scaler.pkl")
    scaler.save(path)
    assert GasolineScaler.load(path).transform_point(rows[0]) == scaler.transform_point(rows[0])


def test_local_loader_checks_missing_artifacts_before_import(monkeypatch, tmp_path):
    monkeypatch.setattr(model_loader, "LOCAL_MODEL_PATH", str(tmp_path / "missing.keras"))
    with pytest.raises(FileNotFoundError):
        model_loader._load_from_local()


def _csv(rows, fieldnames):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _bad_uploads():
    rows = list(csv.DictReader(io.StringIO(SAMPLE.read_text())))
    fields = list(rows[0].keys())
    negative = [dict(r) for r in rows]
    negative[5]["gasoline_price"] = "-1"
    return {
        "missing_column": _csv(rows, [f for f in fields if f != "gasoline_price"]).encode(),
        "too_few_rows": _csv(rows[:40], fields).encode(),
        "non_positive": _csv(negative, fields).encode(),
        "not_utf8": _csv(rows, fields).encode("utf-16"),
    }


@pytest.mark.parametrize("case", ["missing_column", "too_few_rows", "non_positive", "not_utf8"])
def test_invalid_upload_rejected_with_400(client, monkeypatch, tmp_path, case):
    monkeypatch.setattr(data_router, "UPLOAD_DIR", str(tmp_path))
    body = _bad_uploads()[case]
    response = client.post("/data/upload", files={"file": ("bad.csv", body)})
    assert response.status_code == 400
    assert response.json()["detail"]
    assert not list(tmp_path.iterdir())
