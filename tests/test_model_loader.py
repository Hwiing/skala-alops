"""MLflow Production 로더와 캐시 교체(reload_model) 테스트.

실제 MLflow·TensorFlow 대신 가짜 레지스트리를 넣어 빠르게 검증한다.
"""

import sys
from types import SimpleNamespace

import pytest

from serving_app import model_loader


@pytest.fixture
def registry(monkeypatch):
    """state["production"]이 현재 Production 버전 번호(없으면 None)인 가짜 레지스트리."""
    state = {"production": "3", "loaded_uris": [], "fail_load": False}

    class FakeClient:
        def get_latest_versions(self, name, stages):
            assert name == "GasolinePricePredictor"
            assert stages == ["Production"]
            v = state["production"]
            return [SimpleNamespace(version=v)] if v else []

    def load_model(uri):
        if state["fail_load"]:
            raise OSError("artifact download failed")
        state["loaded_uris"].append(uri)
        return f"keras:{uri}"

    monkeypatch.setattr("mlflow.tracking.MlflowClient", FakeClient)
    monkeypatch.setitem(sys.modules, "mlflow.tensorflow", SimpleNamespace(load_model=load_model))
    monkeypatch.setattr(model_loader.GasolineScaler, "load", lambda path: "scaler")
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.setattr(model_loader, "_model_cache", None)
    return state


def test_mlflow_loads_resolved_production_version(registry):
    model = model_loader._load_from_mlflow()
    assert model.version == "production:3"
    assert registry["loaded_uris"] == ["models:/GasolinePricePredictor/3"]


def test_mlflow_without_production_raises_file_not_found(registry):
    registry["production"] = None
    with pytest.raises(FileNotFoundError):
        model_loader._load_from_mlflow()


@pytest.mark.parametrize(
    "after, expected",
    [("2", "production:2"), ("1", "production:1")],
    ids=["gate_passed_promoted", "gate_failed_kept"],
)
def test_reload_reflects_current_production(registry, after, expected):
    registry["production"] = "1"
    model_loader.get_model()
    registry["production"] = after

    result = model_loader.reload_model()

    assert result == {"reloaded": True, "version": expected}
    assert model_loader.get_model().version == expected


def test_reload_failure_keeps_previous_model(registry):
    registry["production"] = "1"
    previous = model_loader.get_model()
    registry["production"] = "2"
    registry["fail_load"] = True

    result = model_loader.reload_model()

    assert result["reloaded"] is False
    assert result["version"] == "production:1"
    assert "artifact download failed" in result["error"]
    assert model_loader.get_model() is previous
