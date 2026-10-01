import csv
import io
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from data.diesel_features import INPUT_DAYS
from data.features import GasolineScaler, build_sequences, load_rows
from serving_app import model_loader
from serving_app.main import app
from serving_app.monitoring import retrain_trigger
from serving_app.monitoring.prediction_window import PredictionWindow
from serving_app.routers import data as data_router
from serving_app.routers import predict as predict_router
from serving_app.schemas import BATCH_MIN_ROWS

SAMPLE = Path("data/sample_diesel_prices.csv")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LOADING_MODE", "lazy")
    monkeypatch.setattr(model_loader, "_model_cache", None)
    with TestClient(app) as client:
        yield client


def make_rows(n, start="2026-01-01"):
    """하루 간격 경유 행. 가격이 하루 1원씩 올라 주간 평균을 손으로 확인하기 쉽다."""
    first = date.fromisoformat(start)
    return [
        {
            "date": (first + timedelta(days=i)).isoformat(),
            "diesel_price": 1500.0 + i,
            "singapore_diesel_price": 90.0,
            "usd_krw": 1400.0,
            "tax_or_supply_feature": 0.0,
        }
        for i in range(n)
    ]


def payload(length=INPUT_DAYS):
    return {"sequence": make_rows(length)}


def test_dashboard_health_and_docs(client):
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200
    health = client.get("/health").json()
    assert health["model_loaded"] is False
    assert health["model_version"] is None
    assert health["model_source"] == "local"


@pytest.mark.parametrize("length", [INPUT_DAYS - 1, INPUT_DAYS + 1])
def test_invalid_sequence_length(client, length):
    assert client.post("/predict", json=payload(length)).status_code == 422


@pytest.mark.parametrize("value", [0, -1, "NaN", "Infinity"])
def test_invalid_price(client, value):
    body = payload()
    body["sequence"][0]["diesel_price"] = value
    assert client.post("/predict", json=body).status_code == 422


@pytest.mark.parametrize("case", ["missing", "duplicate", "gap"])
def test_invalid_dates_rejected(client, case):
    body = payload()
    seq = body["sequence"]
    if case == "missing":
        del seq[5]["date"]
    elif case == "duplicate":
        seq[5]["date"] = seq[4]["date"]
    else:
        seq[5]["date"] = seq[6]["date"]
    assert client.post("/predict", json=body).status_code == 422


def test_predict_returns_four_weeks_with_dates(client, monkeypatch):
    class StubModel:
        version = "champion:3"

        def predict(self, rows):
            assert len(rows) == INPUT_DAYS
            assert rows[-1]["date"] == "2026-04-30"
            return [1601.0, 1602.0, 1603.0, 1604.456]

    monkeypatch.setattr(model_loader, "_model_cache", StubModel())
    response = client.post("/predict", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert body["base_date"] == "2026-04-30"
    assert body["model_version"] == "champion:3"
    assert body["predictions"][0] == {
        "horizon_week": 1,
        "start_date": "2026-05-01",
        "end_date": "2026-05-07",
        "predicted_avg_price": 1601.0,
    }
    assert body["predictions"][3] == {
        "horizon_week": 4,
        "start_date": "2026-05-22",
        "end_date": "2026-05-28",
        "predicted_avg_price": 1604.46,
    }
    assert client.get("/health").json()["model_version"] == "champion:3"


def test_missing_model_returns_503(client, monkeypatch):
    def missing():
        raise FileNotFoundError("test missing model")

    monkeypatch.setattr(model_loader, "get_model", missing)
    assert client.post("/predict", json=payload()).status_code == 503


def _public(record):
    """내부 기록 → HTTP BatchPair 필드만."""
    return {k: record[k] for k in ("date", "predicted", "actual", "naive")}


class LastPriceModel:
    """마지막 입력일 가격 + k를 k주 예측으로 돌려주는 가짜 모델 (짝 정렬을 확인하기 쉽게)."""

    version = "champion:1"

    def predict(self, rows):
        assert len(rows) == INPUT_DAYS
        return [rows[-1]["diesel_price"] + k for k in range(1, 5)]


@pytest.fixture
def batch(client, monkeypatch):
    """가짜 모델·트리거·reload로 batch_test를 돌리는 환경. state로 결과를 조작한다."""
    state = {"trigger": {"status": "ok"}, "reload": {"reloaded": True, "version": "champion:2"}}
    calls = {"trigger": [], "reload": []}

    def trigger(recent):
        calls["trigger"].append(list(recent))
        if state["trigger"] is NotImplementedError:
            raise NotImplementedError("TODO")
        return dict(state["trigger"])

    def reload(expected_version):
        calls["reload"].append(expected_version)
        return dict(state["reload"])

    monkeypatch.setattr(model_loader, "_model_cache", LastPriceModel())
    monkeypatch.setattr(retrain_trigger, "check_and_trigger", trigger)
    monkeypatch.setattr(model_loader, "reload_model", reload)
    monkeypatch.setattr(predict_router, "recent_predictions", PredictionWindow())

    def post(length=BATCH_MIN_ROWS):
        return client.post("/predict/batch-test", json={"rows": make_rows(length)})

    return SimpleNamespace(post=post, state=state, calls=calls)


def test_batch_pairs_each_base_date_with_weekly_actuals(batch):
    response = batch.post()

    assert response.status_code == 200
    pairs = response.json()["predictions"]
    assert len(pairs) == 28
    # 첫 기준일 = 120번째 행(가격 1619). k주 실제 = 기준일 뒤 7일 평균 → 1619 + 7(k−1) + 4
    assert pairs[0] == {
        "date": "2026-04-30",
        "predicted": [1620.0, 1621.0, 1622.0, 1623.0],
        "actual": [1623.0, 1630.0, 1637.0, 1644.0],
        "naive": 1619.0,
    }
    assert pairs[-1]["date"] == "2026-05-27"
    # 내부 기록에는 모델 버전·출처가 붙고, AIOps에는 그 기록이 그대로 넘어간다
    recorded = predict_router.recent_predictions.pairs()
    assert [_public(p) for p in recorded] == pairs
    assert {(p["model_version"], p["source"]) for p in recorded} == {("champion:1", "batch")}
    assert batch.calls["trigger"] == [recorded]


def test_batch_rejects_too_few_rows(batch):
    assert batch.post(BATCH_MIN_ROWS - 1).status_code == 422


def test_batch_resend_does_not_double_count_base_dates(batch):
    batch.post(BATCH_MIN_ROWS + 5)  # 기준일 33개
    batch.post()  # 그중 앞 28개 기준일을 다시 보냄

    assert len(predict_router.recent_predictions) == 33  # 같은 기준일은 덮어씀


@pytest.mark.parametrize("registry_version", ["2", 2])
def test_batch_promoted_reloads_promoted_version_and_clears_window(batch, registry_version):
    batch.state["trigger"] = {
        "status": "promoted",
        "promoted": True,
        "version": registry_version,
        "production_before": 1,
        "rmse": [1.0] * 4,
        "naive_rmse": [2.0] * 4,
    }

    response = batch.post()

    assert response.status_code == 200
    assert response.json()["drift_check"]["version"] == "2"
    assert response.json()["drift_check"]["production_before"] == "1"
    assert response.json()["drift_check"]["reload"] == {"reloaded": True, "version": "champion:2"}
    assert batch.calls["reload"] == ["2"]
    assert len(predict_router.recent_predictions) == 0


def test_batch_reload_failure_keeps_window(batch):
    batch.state["trigger"] = {
        "status": "promoted",
        "promoted": True,
        "version": "2",
        "rmse": [1.0] * 4,
        "naive_rmse": [2.0] * 4,
    }
    batch.state["reload"] = {"reloaded": False, "version": "champion:1", "error": "boom"}

    response = batch.post()

    assert response.json()["drift_check"]["reload"]["reloaded"] is False
    assert len(predict_router.recent_predictions) == 28


def test_batch_not_promoted_skips_reload(batch):
    batch.state["trigger"] = {
        "status": "gate_failed",
        "promoted": False,
        "rmse": [3.0] * 4,
        "naive_rmse": [2.0] * 4,
    }

    response = batch.post()

    assert "reload" not in response.json()["drift_check"]
    assert batch.calls["reload"] == []
    assert len(predict_router.recent_predictions) == 28


def test_batch_unimplemented_trigger_returns_501(batch):
    batch.state["trigger"] = NotImplementedError

    assert batch.post().status_code == 501


def test_batch_missing_model_returns_503(batch, monkeypatch):
    def missing():
        raise FileNotFoundError("test missing model")

    monkeypatch.setattr(model_loader, "get_model", missing)
    assert batch.post().status_code == 503


def _csv(rows, fieldnames=None):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames or list(rows[0]), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def test_csv_upload_and_status(client, monkeypatch, tmp_path):
    monkeypatch.setattr(data_router, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(data_router, "latest_upload", lambda: str(next(tmp_path.glob("*.csv"))))
    body = _csv(make_rows(BATCH_MIN_ROWS))
    response = client.post("/data/upload", files={"file": ("rows.csv", body)})
    assert response.status_code == 200
    assert response.json()["rows"] == BATCH_MIN_ROWS
    status = client.get("/data/status").json()
    assert status["rows"] == BATCH_MIN_ROWS
    assert status["min_price"] > 0


def _bad_uploads():
    rows = make_rows(BATCH_MIN_ROWS)
    fields = list(rows[0])
    negative = [dict(r) for r in rows]
    negative[5]["diesel_price"] = -1
    duplicate = [dict(r) for r in rows]
    duplicate[3]["date"] = duplicate[2]["date"]
    return {
        "missing_column": _csv(rows, [f for f in fields if f != "diesel_price"]).encode(),
        "too_few_rows": _csv(rows[:-1]).encode(),
        "non_positive": _csv(negative).encode(),
        "duplicate_date": _csv(duplicate).encode(),
        "not_utf8": _csv(rows).encode("utf-16"),
    }


@pytest.mark.parametrize(
    "case", ["missing_column", "too_few_rows", "non_positive", "duplicate_date", "not_utf8"]
)
def test_invalid_upload_rejected_with_400(client, monkeypatch, tmp_path, case):
    monkeypatch.setattr(data_router, "UPLOAD_DIR", str(tmp_path))
    body = _bad_uploads()[case]
    response = client.post("/data/upload", files={"file": ("bad.csv", body)})
    assert response.status_code == 400
    assert response.json()["detail"]
    assert not list(tmp_path.iterdir())


def test_sequences_align_next_day_and_scaler_roundtrip(tmp_path):
    rows = load_rows(str(SAMPLE))
    scaler = GasolineScaler().fit(rows[:80])
    X, y = build_sequences(rows, scaler)
    assert len(X) == len(y) == 100
    assert len(X[0]) == 20 and len(X[0][0]) == 4
    assert y[0] == rows[20]["diesel_price"]
    for row in (rows[0], rows[-1]):
        price = row["diesel_price"]
        assert scaler.inverse_price(scaler.scale_price(price)) == pytest.approx(price)
    path = str(tmp_path / "scaler.pkl")
    scaler.save(path)
    assert GasolineScaler.load(path).transform_point(rows[0]) == scaler.transform_point(rows[0])


def test_local_loader_checks_missing_artifacts_before_import(monkeypatch, tmp_path):
    monkeypatch.setattr(model_loader, "LOCAL_PYFUNC_PATH", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        model_loader._load_from_local()


# ---------- 실시간 예측 기록 → 업로드로 지연 정답 채우기 → 판정 (#14) ----------


def _upload(client, monkeypatch, tmp_path, rows):
    monkeypatch.setattr(data_router, "UPLOAD_DIR", str(tmp_path))
    return client.post("/data/upload", files={"file": ("rows.csv", _csv(rows))})


def test_predict_records_live_prediction_without_actual(batch, client):
    client.post("/predict", json=payload())  # 마지막 입력일 2026-04-30, 가격 1619

    assert predict_router.recent_predictions.pairs() == [
        {
            "date": "2026-04-30",
            "predicted": [1620.0, 1621.0, 1622.0, 1623.0],
            "actual": [None] * 4,
            "naive": 1619.0,
            "model_version": "champion:1",
            "source": "live",
        }
    ]
    assert batch.calls["trigger"] == []


def test_upload_fills_only_fully_covered_weeks_and_judges(batch, client, monkeypatch, tmp_path):
    client.post("/predict", json=payload())
    # 2025-11-20 ~ 2026-05-13: 1주차(05-01~05-07)만 7일이 모두 있고 2주차(05-08~05-14)는 05-14가 없다
    rows = make_rows(BATCH_MIN_ROWS, start="2025-11-20")
    prices = {r["date"]: r["diesel_price"] for r in rows}
    week1 = sum(prices[f"2026-05-0{d}"] for d in range(1, 8)) / 7

    response = _upload(client, monkeypatch, tmp_path, rows)

    assert response.status_code == 200
    body = response.json()
    assert body["filled"] == 1
    assert body["drift_check"]["status"] == "ok"
    recorded = predict_router.recent_predictions.pairs()
    assert recorded[0]["actual"] == [round(week1, 2), None, None, None]
    assert batch.calls["trigger"] == [recorded]


def test_upload_without_matured_live_predictions_skips_judgement(
    batch, client, monkeypatch, tmp_path
):
    body = _upload(client, monkeypatch, tmp_path, make_rows(BATCH_MIN_ROWS)).json()

    assert body["filled"] == 0
    assert "drift_check" not in body
    assert batch.calls["trigger"] == []


def test_upload_promotion_reloads_and_clears_window(batch, client, monkeypatch, tmp_path):
    client.post("/predict", json=payload())
    batch.state["trigger"] = {
        "status": "promoted",
        "promoted": True,
        "version": "2",
        "rmse": [1.0] * 4,
        "naive_rmse": [2.0] * 4,
    }

    body = _upload(client, monkeypatch, tmp_path, make_rows(BATCH_MIN_ROWS)).json()

    assert body["filled"] == 4
    assert body["drift_check"]["reload"] == {"reloaded": True, "version": "champion:2"}
    assert batch.calls["reload"] == ["2"]
    assert len(predict_router.recent_predictions) == 0
