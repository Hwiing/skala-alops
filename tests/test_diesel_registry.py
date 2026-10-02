"""레지스트리 버전 번호 형식. MlflowClient(sqlite)는 version을 int로 돌려준다.

FineTuneResult·DriftCheck는 version을 문자열로 받으므로 int가 새면 재학습 결과가 retrain_failed가 된다.
mlflow·tensorflow가 필요한 모듈이라 requirements-dev만 설치한 CI에서는 건너뛴다.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("tensorflow")

from serving_app import diesel_registry  # noqa: E402


def test_production_version_is_string():
    client = SimpleNamespace(get_latest_versions=lambda name, stages: [SimpleNamespace(version=2)])
    assert diesel_registry.production_version(client) == "2"


def test_fine_tune_records_actual_learning_rate(monkeypatch):
    """환경변수로 바꾼 학습률·epoch가 실제 학습과 MLflow 기록(meta)에 그대로 전달된다."""
    from data.diesel_features import FINETUNE_MIN_ROWS
    from tests.test_diesel import make_rows

    seen = {}

    def fake_finetune(forecaster, frame, train_idx, epochs, lr):
        seen["train"] = (epochs, lr)
        return {
            "train_target_period": ["a", "b"],
            "train_windows": 1,
            "seeds": [0],
            "epochs": [epochs],
            "learning_rate": lr,
        }

    def fake_log_and_gate(forecaster, frame, val_idx, meta, run_name):
        seen["meta"] = meta
        return {
            "rmse": [1.0] * 4,
            "naive_rmse": [2.0] * 4,
            "production_rmse": [1.0] * 4,
            "production_before": "1",
            "passed": False,
            "reasons": ["x"],
            "promoted": False,
        }

    monkeypatch.setattr(diesel_registry, "FINE_TUNE_LR", 5e-4)
    monkeypatch.setattr(diesel_registry, "FINE_TUNE_EPOCHS", 20)
    monkeypatch.setattr(diesel_registry, "production_version", lambda client: "1")
    monkeypatch.setattr(diesel_registry, "MlflowClient", lambda: None)
    monkeypatch.setattr(diesel_registry, "load_forecaster", lambda v: object())
    monkeypatch.setattr(diesel_registry, "finetune", fake_finetune)
    monkeypatch.setattr(diesel_registry, "evaluate", lambda f, fr, idx: {"rmse": [1.0] * 4})
    monkeypatch.setattr(diesel_registry, "report", lambda meta: "")
    monkeypatch.setattr(diesel_registry, "log_and_gate", fake_log_and_gate)

    diesel_registry.fine_tune(make_rows(n=FINETUNE_MIN_ROWS + 10))

    assert seen["train"] == (20, 5e-4)
    assert seen["meta"]["learning_rate"] == 5e-4
    assert seen["meta"]["epochs"] == [20]
    assert seen["meta"]["base_version"] == "1"


def test_initial_training_only_logs_and_never_evaluates_or_promotes(monkeypatch):
    meta = {"rmse": [100.0] * 4, "naive_rmse": [2.0] * 4}
    forecaster = object()
    seen = {}
    monkeypatch.setattr(
        diesel_registry, "_fit_initial", lambda *args: (forecaster, object(), [1], meta)
    )
    monkeypatch.setattr(
        diesel_registry.mlflow,
        "start_run",
        lambda **kwargs: nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="initial-run"))),
    )
    monkeypatch.setattr(diesel_registry.mlflow, "set_tags", lambda tags: seen.update(tags))
    monkeypatch.setattr(diesel_registry, "_log_training", lambda model, info: "models:/logged-only")
    monkeypatch.setattr(
        diesel_registry,
        "check_gate",
        lambda *args: pytest.fail("initial training evaluated a gate"),
    )
    monkeypatch.setattr(
        diesel_registry, "MlflowClient", lambda: pytest.fail("initial training accessed registry")
    )
    result = diesel_registry.train_initial("uploaded.csv")
    assert result == {
        "rmse": [100.0] * 4,
        "naive_rmse": [2.0] * 4,
        "run_id": "initial-run",
        "model_uri": "models:/logged-only",
    }
    assert seen == {"workflow": "initial-training", "training_only": "true"}
