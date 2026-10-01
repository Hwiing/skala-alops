"""
경유 1~4주 평균 예측 LSTM과 예측기(contracts.md v2).

구조: LSTM(32) → Dense(16, relu) → Dense(4). 입력 (28일, 8피처), 출력 = 1~4주 평균 세전 가격 변화(표준화).
seed 여러 개의 평균을 쓴다(단일 seed 편차 완화). 정책 규칙은 DieselForecaster.predict()에서 적용해
서빙은 최근 120일 행만 넘기면 4개 가격을 받는다.
"""

import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from tensorflow import keras

from data.diesel_features import FEATURES, HORIZONS, INPUT_DAYS, SEQ_LEN, DailyFrame, FeatureScaler

MODEL_DIR = "serving_app/models/diesel"


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(SEQ_LEN, len(FEATURES))),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(HORIZONS),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def train(
    X: np.ndarray, Y: np.ndarray, weights: np.ndarray, seed: int, epochs: int = 200
) -> tuple[keras.Model, int]:
    """시간순 마지막 10%로 조기 종료(patience 15). 반환: (모델, 실제 학습 epoch 수)."""
    keras.utils.set_random_seed(seed)
    model = build_model()
    stop = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    hist = model.fit(
        X,
        Y,
        sample_weight=weights,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=0,
        callbacks=[stop],
    )
    return model, len(hist.history["loss"])


class DieselForecaster:
    """seed별 LSTM + 학습 구간 scaler + 정책 규칙. predict(rows)는 최근 120일 행으로 1~4주 평균가를 낸다."""

    def __init__(self, models: list[keras.Model], scaler: FeatureScaler):
        self.models, self.scaler = models, scaler

    def predict_changes(self, X: np.ndarray) -> np.ndarray:
        """표준화된 입력 (N, 28, 8) → 원/L 세전 가격 변화 (N, 4), seed 평균."""
        out = np.mean([m.predict(X, verbose=0) for m in self.models], axis=0)
        return out * np.array(self.scaler.y_std)

    def predict(self, rows: list[dict]) -> list[dict]:
        if len(rows) < INPUT_DAYS:
            raise ValueError(f"최근 {INPUT_DAYS}일 행이 필요합니다 (현재 {len(rows)}행)")
        frame = DailyFrame(rows[-INPUT_DAYS:])
        i = len(frame.dates) - 1
        seq = [frame.features(j) for j in range(i - SEQ_LEN + 1, i + 1)]
        X = np.array([self.scaler.transform(seq)], dtype="float32")
        prices = frame.apply_policy(i, list(self.predict_changes(X)[0]))
        base = frame.dates[i]
        return [
            {
                "horizon_week": k + 1,
                "start_date": (base + timedelta(days=7 * k + 1)).isoformat(),
                "end_date": (base + timedelta(days=7 * k + 7)).isoformat(),
                "predicted_avg_price": round(p, 1),
            }
            for k, p in enumerate(prices)
        ]

    def save(self, path: str = MODEL_DIR, meta: dict | None = None):
        Path(path).mkdir(parents=True, exist_ok=True)
        for k, m in enumerate(self.models):
            m.save(f"{path}/lstm_seed{k}.keras")
        with open(f"{path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler.__dict__, f)
        Path(f"{path}/meta.json").write_text(
            json.dumps(
                {"saved": date.today().isoformat(), **(meta or {})}, ensure_ascii=False, indent=2
            )
        )

    @classmethod
    def load(cls, path: str = MODEL_DIR) -> "DieselForecaster":
        scaler = FeatureScaler()
        with open(f"{path}/scaler.pkl", "rb") as f:
            scaler.__dict__.update(pickle.load(f))
        models = [keras.models.load_model(p) for p in sorted(Path(path).glob("lstm_seed*.keras"))]
        return cls(models, scaler)
