"""레지스트리 버전 번호 형식. MlflowClient(sqlite)는 version을 int로 돌려준다.

FineTuneResult·DriftCheck는 version을 문자열로 받으므로 int가 새면 재학습 결과가 retrain_failed가 된다.
mlflow·tensorflow가 필요한 모듈이라 requirements-dev만 설치한 CI에서는 건너뛴다.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("tensorflow")

from serving_app import diesel_registry  # noqa: E402


def test_production_version_is_string():
    client = SimpleNamespace(get_latest_versions=lambda name, stages: [SimpleNamespace(version=2)])
    assert diesel_registry.production_version(client) == "2"
