"""MLflow champion 로더와 캐시 교체(reload_model) 테스트.

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
    """state["champion"]이 현재 champion alias가 가리키는 버전 번호(없으면 None)인 가짜 레지스트리."""
    state = {"champion": "3", "loaded_uris": [], "fail_load": False}

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            assert (name, alias) == ("DieselPricePredictor", "champion")
            if state["champion"] is None:
                raise MlflowException(f"alias {alias} not found")
            return SimpleNamespace(version=state["champion"])

    def load_model(uri):
        if state["fail_load"]:
            raise OSError("artifact download failed")
        state["loaded_uris"].append(uri)
        return f"pyfunc:{uri}"

    fakes = {
        "mlflow": SimpleNamespace(set_tracking_uri=lambda uri: None),
        "mlflow.exceptions": SimpleNamespace(MlflowException=MlflowException),
        "mlflow.tracking": SimpleNamespace(MlflowClient=FakeClient),
        "mlflow.pyfunc": SimpleNamespace(load_model=load_model),
    }
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.setattr(model_loader, "_model_cache", None)
    return state


def test_mlflow_loads_version_resolved_from_champion(registry):
    model = model_loader._load_from_mlflow()
    assert model.version == "champion:3"
    assert model.registry_version == "3"
    assert registry["loaded_uris"] == ["models:/DieselPricePredictor/3"]


def test_mlflow_without_champion_raises_file_not_found(registry):
    """새 mlflow.db처럼 alias가 없을 때도 /predict가 503이 되도록 한다."""
    registry["champion"] = None
    with pytest.raises(FileNotFoundError):
        model_loader._load_from_mlflow()


def test_reload_serves_promoted_version(registry):
    registry["champion"] = "1"
    model_loader.get_model()
    registry["champion"] = "2"

    result = model_loader.reload_model("2")

    assert result == {"reloaded": True, "version": "champion:2"}
    assert model_loader.get_model().version == "champion:2"


def test_reload_rejects_when_champion_is_not_promoted_version(registry):
    registry["champion"] = "1"
    previous = model_loader.get_model()

    result = model_loader.reload_model("2")

    assert result["reloaded"] is False
    assert result["version"] == "champion:1"
    assert "v1" in result["error"] and "v2" in result["error"]
    assert model_loader.get_model() is previous


def test_reload_failure_keeps_previous_model(registry):
    registry["champion"] = "1"
    previous = model_loader.get_model()
    registry["champion"] = "2"
    registry["fail_load"] = True

    result = model_loader.reload_model("2")

    assert result["reloaded"] is False
    assert result["version"] == "champion:1"
    assert "artifact download failed" in result["error"]
    assert model_loader.get_model() is previous


@pytest.mark.parametrize("expected", ["2", None], ids=["local_source", "no_version"])
def test_reload_never_reports_unverified_swap(monkeypatch, expected):
    """local 모드는 레지스트리 버전을 서빙하지 않고, 버전 없는 승격은 확인할 수 없다 → 교체 안 함."""
    previous = SimpleNamespace(version="local")
    monkeypatch.setattr(model_loader, "_model_cache", previous)
    monkeypatch.setenv("MODEL_SOURCE", "local" if expected else "mlflow")

    result = model_loader.reload_model(expected)

    assert result["reloaded"] is False
    assert result["version"] == "local"
    assert result["error"]
    assert model_loader._model_cache is previous
