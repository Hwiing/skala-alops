"""
Day2: MLflow로 휘발유 LSTM 모델을 학습 -> 기록(Tracking) -> 게이트 검증 -> 등록(Registry) -> Production 승격.
Day3: 드리프트 감지 후 Production 가중치에서 이어서 학습하는 fine-tuning 재학습.

실습 시나리오 (94번 슬라이드를 LSTM 버전으로 재구성):
    1) 휘발유 데이터로 base 모델 학습(50 epoch) -> RMSE 확인 (게이트 미달 가능)
    2) 게이트(10원/L) 통과 시 Production으로 승격
    3) (Day3) 드리프트 감지 시 Production 가중치에서 warm-start -> 최근 1개월 데이터로
       10 epoch만 fine-tuning (처음부터 다시 학습하지 않음 - 30일로는 스크래치 학습이 불안정)

실행:
    (대시보드에서 휘발유 CSV를 먼저 업로드하세요 - data/sample_diesel_prices.csv가 예시입니다)
    python scripts/train_baseline_v1.py     # 최초 1회 (scaler.pkl 생성)
    python serving_app/train_and_register.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import mlflow
import mlflow.tensorflow
import numpy as np
from mlflow.tracking import MlflowClient
from tensorflow import keras

from data.features import SEQ_LEN, GasolineScaler, build_sequences, load_rows, train_test_split
from data.storage import latest_upload
from serving_app.lstm_model import build_model

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

# 시드 고정: LSTM 가중치 초기화가 랜덤이라 시드 없이는 실행마다 RMSE가 크게 흔들려
# (관찰치: 2.22~5.29) 게이트(10원/L) 통과 여부가 운에 좌우됩니다. numpy/tensorflow/python
# random을 한 번에 고정해 재현 가능한 학습 결과를 보장합니다.
SEED = 42
keras.utils.set_random_seed(SEED)

RMSE_GATE = 10.00
MODEL_NAME = "GasolinePricePredictor"
SCALER_PATH = "serving_app/models/scaler.pkl"
BASE_EPOCHS = 100  # 3층 LSTM + 3년치 데이터 기준, RMSE가 안정적으로 게이트 아래로 수렴하는 지점
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-4  # base 학습(1e-3)보다 낮은 학습률로 살짝만 갱신


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def _prepare(rows: list[dict], scaler: GasolineScaler):
    X, y = build_sequences(rows, scaler)
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")
    y_train_scaled = np.array([scaler.scale_price(v) for v in y_train], dtype="float32")
    return X_train, y_train_scaled, X_test, y_test


def _register_if_gate_passed(model, run_id: str, score: float, naive_score: float) -> dict:
    result = {"run_id": run_id, "rmse": score, "naive_rmse": naive_score, "promoted": False}
    if (
        math.isfinite(score)
        and math.isfinite(naive_score)
        and score <= RMSE_GATE
        and score < naive_score
    ):
        v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        MlflowClient().transition_model_version_stage(
            name=MODEL_NAME, version=v.version, stage="Production"
        )
        result["promoted"] = True
        result["version"] = v.version
        print(f"[GATE PASSED] rmse={score:.2f} -> {MODEL_NAME} v{v.version} promoted to Production")
    else:
        print(
            f"[GATE FAILED] rmse={score:.2f} naive_rmse={naive_score:.2f}, gate<={RMSE_GATE} AND <naive 미충족 -> 배포 차단, 기존 Production 유지"
        )
    return result


def train_and_register(csv_path: str | None = None, rows: list[dict] | None = None) -> dict:
    """Day2: 처음부터(scratch) 학습. 데이터가 충분한 base 학습에서만 사용합니다.

    csv_path를 지정하지 않으면 data/uploads/에 가장 최근 업로드된 CSV를 사용합니다
    (data/storage.py의 latest_upload() - 대시보드에서 업로드한 파일).
    """
    if rows is None:
        rows = load_rows(csv_path or latest_upload())
    scaler = GasolineScaler.load(SCALER_PATH)
    X_train, y_train_scaled, X_test, y_test = _prepare(rows, scaler)

    with mlflow.start_run(run_name="base-train"):
        model = build_model()
        model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, verbose=0)

        preds = [scaler.inverse_price(p) for p in model.predict(X_test, verbose=0).flatten()]
        score = rmse(y_test, preds)
        # 동일한 검증 타깃의 직전 날 가격을 naive 예측으로 사용.
        naive_preds = [
            rows[i - 1]["diesel_price"] for i in range(len(rows) - len(y_test), len(rows))
        ]
        naive_score = rmse(y_test, naive_preds)

        mlflow.log_param("mode", "scratch")
        mlflow.log_param("epochs", BASE_EPOCHS)
        mlflow.log_metric("rmse", score)
        mlflow.log_metric("naive_rmse", naive_score)
        mlflow.log_param("sequence_length", SEQ_LEN)
        mlflow.log_artifact(SCALER_PATH, artifact_path="preprocessing")
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        return _register_if_gate_passed(model, mlflow.active_run().info.run_id, score, naive_score)


def fine_tune(rows: list[dict]) -> dict:
    """
    Day3: 현재 Production 모델 가중치에서 이어서(warm start), 넘겨받은 rows(최근 데이터)로
    짧게 fine-tuning합니다. rows가 적을 때(예: 최근 1개월)도 스크래치 학습보다 훨씬 안정적입니다.
    """
    scaler = GasolineScaler.load(SCALER_PATH)
    X_train, y_train_scaled, X_test, y_test = _prepare(rows, scaler)

    model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")

    with mlflow.start_run(run_name="fine-tune"):
        model.fit(X_train, y_train_scaled, epochs=FINE_TUNE_EPOCHS, verbose=0)

        preds = [scaler.inverse_price(p) for p in model.predict(X_test, verbose=0).flatten()]
        score = rmse(y_test, preds)
        # 동일한 검증 타깃의 직전 날 가격을 naive 예측으로 사용.
        naive_preds = [
            rows[i - 1]["diesel_price"] for i in range(len(rows) - len(y_test), len(rows))
        ]
        naive_score = rmse(y_test, naive_preds)

        mlflow.log_param("mode", "fine-tune")
        mlflow.log_param("epochs", FINE_TUNE_EPOCHS)
        mlflow.log_param("n_rows", len(rows))
        mlflow.log_metric("rmse", score)
        mlflow.log_metric("naive_rmse", naive_score)
        mlflow.log_param("sequence_length", SEQ_LEN)
        mlflow.log_artifact(SCALER_PATH, artifact_path="preprocessing")
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        return _register_if_gate_passed(model, mlflow.active_run().info.run_id, score, naive_score)


if __name__ == "__main__":
    train_and_register()
