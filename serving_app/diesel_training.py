"""
경유 LSTM 학습·평가 공용 로직. baseline 스크립트(#9), MLflow 등록(#10), fine-tuning(#11)이 같이 쓴다.

- 검증: 정답이 확보된 마지막 holdout_days일. 학습 타깃이 검증 날짜와 겹치지 않게 그 앞 28일은 비운다.
- scaler는 학습 구간에만 fit, 최근 1년 학습 표본 가중치 3배.
- naive: 마지막 입력일 가격을 1~4주 평균 예측으로 사용.
"""

import numpy as np

from data.diesel_features import HORIZONS, DailyFrame, FeatureScaler, build_windows
from serving_app.diesel_model import LEARNING_RATE, DieselForecaster, train

RECENT_WEIGHT = 3.0


def rmse(errors) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def split_holdout(frame: DailyFrame, holdout_days: int) -> tuple[list[int], list[int]]:
    answered = [i for i in range(len(frame.dates)) if frame.targets(i) is not None]
    if len(answered) <= holdout_days + 7 * HORIZONS:
        raise ValueError(
            f"정답이 있는 날이 {len(answered)}일뿐이라 검증 {holdout_days}일을 뗄 수 없습니다"
        )
    val_idx = answered[-holdout_days:]
    return [i for i in answered if i <= val_idx[0] - 7 * HORIZONS - 1], val_idx


def to_arrays(frame: DailyFrame, idx, scaler: FeatureScaler | None = None):
    """(scaler, 표준화 X, 표준화 Y, 쓰인 날짜 인덱스). scaler가 없으면 이 구간으로 fit."""
    X, Y, kept = build_windows(frame, idx)
    scaler = scaler or FeatureScaler().fit(X, Y)
    Xs = np.array([scaler.transform(s) for s in X], dtype="float32")
    Ys = np.array([scaler.scale_y(y) for y in Y], dtype="float32")
    return scaler, Xs, Ys, kept


def recent_weights(frame: DailyFrame, kept: list[int]) -> np.ndarray:
    cutoff = frame.dates[kept[-1]].toordinal() - 365
    return np.array(
        [RECENT_WEIGHT if frame.dates[i].toordinal() >= cutoff else 1.0 for i in kept],
        dtype="float32",
    )


def fit(frame: DailyFrame, train_idx: list[int], seeds: list[int]) -> tuple[DieselForecaster, dict]:
    scaler, X, Y, kept = to_arrays(frame, train_idx)
    models, epochs = [], []
    for seed in seeds:
        m, n = train(X, Y, recent_weights(frame, kept), seed)
        models.append(m), epochs.append(n)
    d = frame.dates
    info = {
        "train_target_period": [d[kept[0]].isoformat(), d[kept[-1]].isoformat()],
        "train_windows": len(kept),
        "seeds": seeds,
        "epochs": epochs,
        "learning_rate": LEARNING_RATE,
    }
    return DieselForecaster(models, scaler), info


def finetune(
    forecaster: DieselForecaster, frame: DailyFrame, train_idx: list[int], epochs: int, lr: float
) -> dict:
    """Production 가중치에서 이어서 학습(warm start). scaler는 Production 것을 그대로 쓴다(재fit 금지)."""
    from tensorflow import keras

    _, X, Y, kept = to_arrays(frame, train_idx, forecaster.scaler)
    for m in forecaster.models:
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
        m.fit(X, Y, epochs=epochs, batch_size=64, verbose=0)
    d = frame.dates
    return {
        "train_target_period": [d[kept[0]].isoformat(), d[kept[-1]].isoformat()],
        "train_windows": len(kept),
        "seeds": list(range(len(forecaster.models))),
        "epochs": [epochs] * len(forecaster.models),
        "learning_rate": lr,
    }


def evaluate(forecaster: DieselForecaster, frame: DailyFrame, val_idx: list[int]) -> dict:
    """정책 규칙까지 적용한 1~4주 RMSE와 같은 날짜의 naive RMSE."""
    _, X, _, kept = to_arrays(frame, val_idx, forecaster.scaler)
    changes = forecaster.predict_changes(X)
    pred = np.array([frame.apply_policy(i, list(c)) for i, c in zip(kept, changes)])
    true = np.array([frame.actual(i) for i in kept])
    naive = np.array([[frame.price[i]] * HORIZONS for i in kept])
    # 참고 기준: 모델 없이 세금·상한 규칙만 적용(변화 0). LSTM 자체의 기여를 따로 보기 위함
    rule = np.array([frame.apply_policy(i, [0.0] * HORIZONS) for i in kept])
    d = frame.dates
    return {
        "validation_period": [d[kept[0]].isoformat(), d[kept[-1]].isoformat()],
        "validation_windows": len(kept),
        "rmse": [rmse(true[:, k] - pred[:, k]) for k in range(HORIZONS)],
        "naive_rmse": [rmse(true[:, k] - naive[:, k]) for k in range(HORIZONS)],
        "rule_rmse": [rmse(true[:, k] - rule[:, k]) for k in range(HORIZONS)],
    }


def report(meta: dict) -> str:
    lines = [
        f"데이터 {meta['data_period']} {meta['rows']}행 ({'합성' if meta.get('synthetic') else '실측'})",
        f"학습 타깃 {meta['train_target_period']} {meta['train_windows']}개 · "
        f"검증 {meta['validation_period']} {meta['validation_windows']}개 · seed {meta['seeds']} epoch {meta['epochs']} lr {meta.get('learning_rate')}",
        f"{'k주차 평균 RMSE 원/L':22}" + "".join(f"{k + 1}주".rjust(9) for k in range(HORIZONS)),
        f"{'naive (마지막 입력일 가격)':22}" + "".join(f"{v:9.1f}" for v in meta["naive_rmse"]),
        f"{'naive + 정책 규칙 (참고)':22}" + "".join(f"{v:9.1f}" for v in meta["rule_rmse"]),
        f"{'LSTM + 정책 규칙':22}" + "".join(f"{v:9.1f}" for v in meta["rmse"]),
    ]
    return "\n".join(lines)
