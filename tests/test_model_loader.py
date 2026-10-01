"""MLflow Production 로더와 캐시 교체(reload_model) 테스트.

실제 MLflow·TensorFlow 대신 가짜 모듈을 sys.modules에 넣는다.
CI는 requirements-dev.txt만 설치하므로 mlflow 없이도 돌아가야 한다.
"""

import sys
from types import SimpleNamespace

import pytest

from serving_app import model_loader


class MlflowException(Exception):
    pass


@pytest.fixture
def registry(monkeypatch):
    """state["production"]이 현재 Production 버전 번호(없으면 None)인 가짜 레지스트리."""
    state = {"production": "3", "loaded_uris": [], "fail_load": False, "unregistered": False}

    class FakeClient:
        def get_latest_versions(self, name, stages):
            assert name == "GasolinePricePredictor"
            assert stages == ["Production"]
            if state["unregistered"]:
                raise MlflowException(f"Registered Model with name={name} not found")
            v = state["production"]
            return [SimpleNamespace(version=v)] if v else []

    def load_model(uri):
        if state["fail_load"]:
            raise OSError("artifact download failed")
        state["loaded_uris"].append(uri)
        return f"keras:{uri}"

    fakes = {
        "mlflow": SimpleNamespace(set_tracking_uri=lambda uri: None),
        "mlflow.exceptions": SimpleNamespace(MlflowException=MlflowException),
        "mlflow.tracking": SimpleNamespace(MlflowClient=FakeClient),
        "mlflow.tensorflow": SimpleNamespace(load_model=load_model),
    }
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)
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


def test_mlflow_unregistered_model_raises_file_not_found(registry):
    """새 mlflow.db처럼 모델 이름이 아예 없을 때도 /predict가 503이 되도록 한다."""
    registry["unregistered"] = True
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
