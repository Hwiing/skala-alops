"""
Day1 -> Day2(MLflow 연동) 확장 파일.

Day1 실습 목표: Lazy Loading vs Eager Loading 두 방식을 직접 구현하고
서버 시작 시간 / 첫 요청 응답 시간을 비교합니다. (44번 슬라이드 결과표 참고)
LSTM은 로컬 pickle 모델보다 로딩 자체가 무거워서, 이 비교가 Day1보다 오히려
더 체감됩니다.

Day2 실습 목표: MODEL_SOURCE=mlflow 로 전환해, 로컬 .keras 파일 대신
MLflow Model Registry의 Production 버전을 로드하도록 확장합니다.
main.py / train_and_register.py 코드는 그대로 두고 이 파일만 손대면 되도록
설계되어 있습니다 - 이것이 "조립 블록" 구조입니다.

스케일러(scaler.pkl)는 Day1~3 내내 동일한 파일을 그대로 재사용합니다
(MODEL_SOURCE와 무관하게 항상 로컬 파일에서 로드) - 정규화 기준이 바뀌면
이미 그 기준으로 학습된 가중치와 어긋나기 때문입니다.

환경변수
    LOADING_MODE = lazy(기본값) | eager
    MODEL_SOURCE = local(기본값, Day1) | mlflow(Day2+)
    MLFLOW_TRACKING_URI = MODEL_SOURCE=mlflow 일 때 필요
"""

import os
import threading
import time

from data.features import GasolineScaler

LOCAL_MODEL_PATH = "serving_app/models/gasoline_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
MODEL_NAME = "GasolinePricePredictor"

_model_cache = None  # Lazy Loading 캐시
_reload_lock = threading.Lock()


class LoadedModel:
    """local .keras와 mlflow 두 소스를 동일한 인터페이스로 감싸는 래퍼."""

    def __init__(self, keras_model, scaler: GasolineScaler, version: str):
        self._keras_model = keras_model
        self.scaler = scaler
        self.version = version

    def predict_one(self, sequence: list[dict]) -> float:
        """
        sequence: [{"gasoline_price": ..., "crude_oil_price": ..., "usd_krw": ..., "tax_or_supply_feature": ...}, ...] 길이 SEQ_LEN, 오래된 날 -> 최근 날 순서.
        """
        import numpy as np

        scaled = [self.scaler.transform_point(p) for p in sequence]
        x = np.array([scaled], dtype="float32")  # (1, SEQ_LEN, 4)
        pred_scaled = float(self._keras_model.predict(x, verbose=0)[0][0])
        return self.scaler.inverse_price(pred_scaled)


def _load_from_local() -> LoadedModel:
    if not os.path.isfile(LOCAL_MODEL_PATH) or not os.path.isfile(SCALER_PATH):
        raise FileNotFoundError("baseline 모델과 scaler를 먼저 생성하세요")
    from tensorflow import keras

    keras_model = keras.models.load_model(LOCAL_MODEL_PATH)
    scaler = GasolineScaler.load(SCALER_PATH)
    return LoadedModel(keras_model=keras_model, scaler=scaler, version="v1-local")


def _load_from_mlflow() -> LoadedModel:
    """Production 버전 번호를 먼저 확인한 뒤 그 번호로 로드한다.

    stage URI(.../Production)를 바로 로드하면 확인과 로드 사이에 승격이 일어날 때
    응답의 버전과 실제 가중치가 어긋날 수 있다. 스케일러는 항상 로컬 파일에서 읽는다.
    """
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tensorflow import load_model
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    try:
        versions = MlflowClient().get_latest_versions(MODEL_NAME, ["Production"])
    except MlflowException as exc:  # 등록된 모델 이름이 아직 없는 새 레지스트리 등
        raise FileNotFoundError(f"{MODEL_NAME}을 레지스트리에서 찾을 수 없습니다") from exc
    if not versions:
        raise FileNotFoundError(f"{MODEL_NAME}에 Production 버전이 없습니다")
    version = versions[0].version
    keras_model = load_model(f"models:/{MODEL_NAME}/{version}")
    scaler = GasolineScaler.load(SCALER_PATH)
    return LoadedModel(keras_model=keras_model, scaler=scaler, version=f"production:{version}")


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


def reload_model() -> dict:
    """AIOps 재학습·승격 후 호출. 새 모델 로드에 성공할 때만 캐시를 교체한다.

    실패하면 기존 모델을 그대로 두고 reloaded=False와 error를 돌려준다.
    """
    global _model_cache
    with _reload_lock:
        try:
            model = _load_model()
        except Exception as exc:
            current = _model_cache.version if _model_cache is not None else None
            return {"reloaded": False, "version": current, "error": str(exc)}
        _model_cache = model
        return {"reloaded": True, "version": model.version}
