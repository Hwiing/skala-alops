"""공통 계약의 계층 간 일치와 잘못된 출력/상태의 실제 API 처리를 검증한다."""

import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from data.contracts import BATCH_MIN_ROWS, FINETUNE_MIN_ROWS, INPUT_DAYS, validate_daily_rows
from data.diesel import validate_diesel_rows
from data.diesel_features import load_diesel_rows
from data.features import validate_rows
from serving_app import model_loader
from serving_app.diesel_gate import check_gate
from serving_app.main import app
from serving_app.monitoring import retrain_trigger
from serving_app.monitoring.prediction_window import PredictionWindow
from serving_app.routers import predict as predict_router
from serving_app.schemas import (
    BatchPair,
    BatchTestRequest,
    DriftCheck,
    FineTuneResult,
    PredictRequest,
    PredictResponse,
)


@pytest.fixture
def batch_rows():
    return json.loads(Path("examples/batch-test.json").read_text())["rows"]


def test_examples_are_valid_requests_and_response():
    PredictRequest.model_validate_json(Path("examples/predict.json").read_text())
    BatchTestRequest.model_validate_json(Path("examples/batch-test.json").read_text())
    PredictResponse.model_validate_json(Path("examples/predict-response.json").read_text())


def test_csv_build_upload_and_training_use_same_validation(batch_rows, tmp_path):
    raw = [{k: str(v) for k, v in row.items()} for row in batch_rows]
    expected = validate_daily_rows(raw)
    assert validate_rows(raw) == validate_diesel_rows(raw) == expected
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=raw[0].keys())
    writer.writeheader()
    writer.writerows(raw)
    path = tmp_path / "rows.csv"
    path.write_text(output.getvalue())
    assert load_diesel_rows(str(path)) == expected
    request = BatchTestRequest(rows=raw)
    assert [p.model_dump(mode="json") for p in request.rows] == expected
    assert INPUT_DAYS == 120 and BATCH_MIN_ROWS == 175 and FINETUNE_MIN_ROWS == 627


@pytest.mark.parametrize("bad_date", ["20260101", "2026-01-01T00:00:00", 1767225600])
def test_csv_and_api_both_reject_non_iso_dates(batch_rows, bad_date):
    batch_rows[0]["date"] = bad_date
    with pytest.raises(ValueError):
        validate_diesel_rows(batch_rows)
    with pytest.raises(ValidationError):
        BatchTestRequest(rows=batch_rows)


@pytest.mark.parametrize(
    "column", ["diesel_price", "singapore_diesel_price", "usd_krw", "tax_or_supply_feature"]
)
def test_csv_and_api_both_reject_non_finite_features(batch_rows, column):
    batch_rows[0][column] = "NaN"
    with pytest.raises(ValueError):
        validate_diesel_rows(batch_rows)
    with pytest.raises(ValidationError):
        BatchTestRequest(rows=batch_rows)


def test_api_rejects_legacy_fields(batch_rows):
    batch_rows[0]["gasoline_price"] = batch_rows[0].pop("diesel_price")
    with pytest.raises(ValidationError):
        BatchTestRequest(rows=batch_rows)


@pytest.mark.parametrize(
    "case", ["three_weeks", "swapped_weeks", "wrong_base", "eight_day_week", "extra_field"]
)
def test_predict_response_rejects_wrong_horizons_and_dates(case):
    body = json.loads(Path("examples/predict-response.json").read_text())
    if case == "three_weeks":
        body["predictions"].pop()
    elif case == "swapped_weeks":
        body["predictions"].reverse()
    elif case == "wrong_base":
        body["base_date"] = "2026-04-29"
    elif case == "eight_day_week":
        body["predictions"][0]["end_date"] = "2026-05-08"
    else:
        body["predictions"][0]["actual"] = 1700
    with pytest.raises(ValidationError):
        PredictResponse.model_validate(body)


@pytest.mark.parametrize(
    "bad", [[1600] * 3, [1600] * 5, [1600, "NaN", 1600, 1600], [0, 1600, 1600, 1600]]
)
@pytest.mark.parametrize("endpoint", ["/predict", "/predict/batch-test"])
def test_invalid_model_output_is_503_and_never_added_to_window(
    monkeypatch, batch_rows, bad, endpoint
):
    monkeypatch.setattr(
        model_loader, "_model_cache", SimpleNamespace(version="local", predict=lambda rows: bad)
    )
    existing = PredictionWindow()
    existing.record("2025-01-01", [1500] * 4, 1500, "local", [1501] * 4, "batch")
    before = existing.pairs()
    monkeypatch.setattr(predict_router, "recent_predictions", existing)
    key, rows = (
        ("sequence", batch_rows[:INPUT_DAYS]) if endpoint == "/predict" else ("rows", batch_rows)
    )
    response = TestClient(app).post(endpoint, json={key: rows})
    assert response.status_code == 503
    assert predict_router.recent_predictions.pairs() == before


@pytest.mark.parametrize(
    "bad",
    [
        {"status": "promoted", "promoted": True, "rmse": [1] * 4, "naive_rmse": [2] * 4},
        {"status": "ok", "promoted": True, "version": "2"},
        {"status": "retrain_triggered", "promoted": False},
        {"status": "gate_failed", "promoted": False, "rmse": [1] * 3, "naive_rmse": [2] * 4},
    ],
)
def test_invalid_aiops_result_never_reloads_or_clears_window(monkeypatch, batch_rows, bad):
    monkeypatch.setattr(
        model_loader,
        "_model_cache",
        SimpleNamespace(version="champion:1", predict=lambda rows: [1600] * 4),
    )
    monkeypatch.setattr(predict_router, "recent_predictions", PredictionWindow())
    monkeypatch.setattr(retrain_trigger, "check_and_trigger", lambda rows: bad)

    def unexpected_reload(version):
        pytest.fail("잘못된 승격 결과는 모델 교체를 호출하면 안 됩니다")

    monkeypatch.setattr(model_loader, "reload_model", unexpected_reload)
    response = TestClient(app).post("/predict/batch-test", json={"rows": batch_rows})
    assert response.status_code == 503
    assert len(predict_router.recent_predictions) == 28


def test_reload_version_must_match_promoted_version():
    with pytest.raises(ValidationError):
        DriftCheck(
            status="promoted",
            promoted=True,
            version="2",
            rmse=[1] * 4,
            naive_rmse=[2] * 4,
            reload={"reloaded": True, "version": "champion:1"},
        )


@pytest.mark.parametrize("status", ["ok", "insufficient_data", "no_production", "retrain_failed"])
def test_non_retrained_status_is_explicit(status):
    assert DriftCheck(status=status).promoted is False


def test_finetune_accepts_weekly_metrics_and_no_production_without_metrics():
    result = FineTuneResult(
        status="promoted", promoted=True, version="3", rmse=[1] * 4, naive_rmse=[2] * 4, passed=True
    )
    assert result.version == "3"
    absent = FineTuneResult(status="no_production", reasons=["Production 없음"])
    assert absent.rmse is None and absent.naive_rmse is None
    with pytest.raises(ValidationError):
        FineTuneResult(status="gate_failed", promoted=False)


@pytest.mark.parametrize(
    "rmse,naive,production", [([1], [2], None), ([1] * 4, [2] * 4, []), ([-1] * 4, [2] * 4, None)]
)
def test_gate_rejects_missing_weeks_and_negative_rmse(rmse, naive, production):
    assert check_gate(rmse, naive, production)["passed"] is False


def test_batch_pair_requires_four_weekly_values_and_positive_naive():
    with pytest.raises(ValidationError):
        BatchPair(date="2026-04-30", predicted=[1600] * 3, actual=[1700] * 4, naive=0)


def test_openapi_exposes_weekly_limits_and_typed_aiops_states():
    schemas = app.openapi()["components"]["schemas"]
    sequence = schemas["PredictRequest"]["properties"]["sequence"]
    assert sequence["minItems"] == sequence["maxItems"] == 120
    assert schemas["BatchTestRequest"]["properties"]["rows"]["minItems"] == 175
    assert schemas["BatchPair"]["properties"]["predicted"]["minItems"] == 4
    assert schemas["DailyPoint"]["additionalProperties"] is False
    assert set(schemas["DriftCheck"]["properties"]["status"]["enum"]) == {
        "ok",
        "insufficient_data",
        "promoted",
        "gate_failed",
        "no_production",
        "retrain_failed",
    }


def test_aiops_insufficient_pairs_never_report_normal():
    from serving_app.monitoring.retrain_trigger import check_and_trigger

    result = check_and_trigger([])
    assert DriftCheck.model_validate(result).status == "insufficient_data"


def test_aiops_window_compares_model_with_same_week_naive(monkeypatch):
    from serving_app.monitoring import drift_detector

    window = [
        {"date": f"2026-04-{d:02d}", "predicted": [1600] * 4, "actual": [1700] * 4, "naive": 1650}
        for d in range(1, 29)
    ]
    calls = []

    def rmse(pairs, *, naive=False):
        calls.append((len(pairs), naive))
        return 15.0 if naive else 16.0  # 하한 10원 위에서 naive보다 나쁨

    monkeypatch.setattr(drift_detector, "compute_rmse", rmse)
    assert drift_detector.is_drift(window)
    assert calls == [(28, False), (28, True)]


def test_aiops_duplicate_base_dates_are_not_recounted():
    from serving_app.monitoring import drift_detector

    one_day = {"date": "2026-04-30", "predicted": [1600] * 4, "actual": [1700] * 4, "naive": 1650}
    result = drift_detector.evaluate([one_day] * 28)  # 같은 기준일 28번 = 짝 1개
    assert result["status"] == "insufficient_data" and result["n"] == 1


def test_health_identifies_active_contract(monkeypatch):
    monkeypatch.setattr(model_loader, "_model_cache", None)
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setenv("LOADING_MODE", "lazy")
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["contract_version"] == "diesel-weekly-v2"
    assert response.json()["model_loaded"] is False
