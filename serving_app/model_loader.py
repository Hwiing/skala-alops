"""
Day1 -> Day2(MLflow 연동) 확장 파일.

Day1 실습 목표: Lazy Loading vs Eager Loading 두 방식을 직접 구현하고
서버 시작 시간 / 첫 요청 응답 시간을 비교합니다. (44번 슬라이드 결과표 참고)
LSTM은 로컬 pickle 모델보다 로딩 자체가 무거워서, 이 비교가 Day1보다 오히려
더 체감됩니다.

Day2 실습 목표: MODEL_SOURCE=mlflow 로 전환해, 로컬 pyfunc 폴더 대신
MLflow Model Registry의 champion alias 버전을 로드하도록 확장합니다.

경유 모델(contracts.md v2)은 pyfunc 하나에 LSTM·scaler·정책 규칙이 모두 들어 있어
서빙은 최근 120일 행을 넘기고 1~4주 평균가 4개를 받기만 합니다.

환경변수
    LOADING_MODE = lazy(기본값) | eager
    MODEL_SOURCE = local(기본값, Day1) | mlflow(Day2+)
    MLFLOW_TRACKING_URI = MODEL_SOURCE=mlflow 일 때 필요
"""

import os
import threading
import time

LOCAL_PYFUNC_PATH = "serving_app/models/diesel_pyfunc"
MODEL_NAME = "DieselPricePredictor"
ALIAS = "champion"

_model_cache = None  # Lazy Loading 캐시
_reload_lock = threading.Lock()


class LoadedModel:
    """local 폴더와 mlflow 레지스트리 pyfunc를 같은 인터페이스로 감싸는 래퍼.

    registry_version: 레지스트리 버전 번호(local은 None). 승격 버전이 실제로 올라왔는지 확인할 때 쓴다.
    """

    def __init__(self, pyfunc, version: str, registry_version: str | None = None):
        self._pyfunc = pyfunc
        self.version = version
        self.registry_version = registry_version

    def predict(self, rows: list[dict]) -> list[float]:
        """rows: 최근 120일 {date, diesel_price, ...}, 오래된 날 -> 최근 날. 반환: 1~4주 평균가 4개."""
        import pandas as pd

        return [float(v) for v in self._pyfunc.predict(pd.DataFrame(rows))]


def _load_from_local() -> LoadedModel:
    if not os.path.isdir(LOCAL_PYFUNC_PATH):
        raise FileNotFoundError(
            f"{LOCAL_PYFUNC_PATH}가 없습니다. diesel_registry.save_local_pyfunc()로 먼저 저장하세요"
        )
    from mlflow.pyfunc import load_model

    return LoadedModel(load_model(LOCAL_PYFUNC_PATH), version="local")


def _load_from_mlflow() -> LoadedModel:
    """champion이 가리키는 버전 번호를 먼저 확인한 뒤 그 번호로 로드한다.

    alias URI(@champion)를 바로 로드하면 확인과 로드 사이에 승격이 일어날 때
    응답의 버전과 실제 가중치가 어긋날 수 있다.
    """
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.pyfunc import load_model
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    try:
        version = str(MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version)
    except MlflowException as exc:  # 등록 모델·alias가 아직 없는 새 레지스트리 등
        raise FileNotFoundError(f"{MODEL_NAME}@{ALIAS}를 레지스트리에서 찾을 수 없습니다") from exc
    pyfunc = load_model(f"models:/{MODEL_NAME}/{version}")
    return LoadedModel(pyfunc, version=f"{ALIAS}:{version}", registry_version=version)


def _load_model() -> LoadedModel:
    source = os.getenv("MODEL_SOURCE", "local")
    if source == "mlflow":
        return _load_from_mlflow()
    return _load_from_local()


def load_eager() -> LoadedModel:
    """Eager Loading: 서버 시작 시점에 즉시 모델을 로드한다."""
    start = time.time()
    model = _load_model()
    print(f"[eager] model loaded in {time.time() - start:.3f}s at startup")
    global _model_cache
    _model_cache = model
    return model


def get_model() -> LoadedModel:
    """Lazy Loading: 첫 요청이 들어올 때만 로드하고, 이후에는 캐시를 재사용한다."""
    global _model_cache
    if _model_cache is None:
        start = time.time()
        _model_cache = _load_model()
        print(f"[lazy] model loaded in {time.time() - start:.3f}s on first request")
    return _model_cache


def reload_model(expected_version: str | None) -> dict:
    """AIOps 재학습·승격 후 호출. 승격된 버전(expected_version)이 실제로 로드될 때만 캐시를 교체한다.

    local 모드·버전 정보 없음·로드 실패·champion이 다른 버전을 가리킴 → 기존 모델 유지,
    reloaded=False와 error를 돌려준다.
    """
    global _model_cache
    with _reload_lock:
        current = _model_cache.version if _model_cache is not None else None

        def keep(reason: str) -> dict:
            return {"reloaded": False, "version": current, "error": reason}

        if expected_version is None:
            return keep("승격 버전 정보가 없어 교체를 확인할 수 없습니다")
        if os.getenv("MODEL_SOURCE", "local") != "mlflow":
            return keep("MODEL_SOURCE=local은 레지스트리 승격 버전을 서빙하지 않습니다")
        try:
            model = _load_from_mlflow()
        except Exception as exc:
            return keep(str(exc))
        if model.registry_version != str(expected_version):
            return keep(
                f"{ALIAS}이 v{model.registry_version}을 가리킵니다 (승격 v{expected_version})"
            )
        _model_cache = model
        return {"reloaded": True, "version": model.version}
