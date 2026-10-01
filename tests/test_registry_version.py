"""MLflow 레지스트리 version 타입 경계: 3.x SQLite 백엔드는 int를 돌려준다 → 레지스트리 경계에서도 str.
(스키마 정규화는 #51 RegistryVersion, tests/test_contracts.py)"""

from mlflow import MlflowClient

from data.contracts import MODEL_NAME
from serving_app.diesel_registry import production_version


def test_real_sqlite_registry_returns_version_as_str(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    client = MlflowClient()
    assert production_version(client) is None  # 등록 모델 없음
    client.create_registered_model(MODEL_NAME)
    created = client.create_model_version(MODEL_NAME, f"file://{tmp_path}/model")
    client.transition_model_version_stage(MODEL_NAME, created.version, "Production")
    assert production_version(client) == "1"  # 실제 반환은 int 1
