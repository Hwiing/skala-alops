"""
#9 경유 LSTM baseline 학습 + naive 비교 (contracts.md v2).

- 검증: 시간순 마지막 --holdout-days일(정답 확보된 날). 학습 타깃이 검증 날짜와 겹치지 않게 그 앞 28일은 비운다.
- scaler: 학습 구간에만 fit. 최근 1년 학습 표본 가중치 3배.
- naive: 마지막 입력일 가격을 1~4주 평균 예측으로 사용.
- 출력: 주차별 RMSE 비교표, serving_app/models/diesel/ (seed별 모델, scaler, meta.json + 리포트)

실행: python scripts/train_diesel_baseline.py [--csv data/processed/diesel_features_2008_spliced.csv]
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")  # 작은 LSTM은 단일 스레드가 더 빠름
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from data.diesel_features import (
    HORIZONS,
    DailyFrame,
    FeatureScaler,
    build_windows,
    load_diesel_rows,
)  # noqa: E402
from serving_app.diesel_model import MODEL_DIR, DieselForecaster, train  # noqa: E402


def rmse(errors) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", default="data/processed/diesel_features_2008_spliced.csv"
    )  # 데이터 담당 PR #32 산출물
    ap.add_argument("--holdout-days", type=int, default=365)
    ap.add_argument("--seeds", default="42,7,2026")
    ap.add_argument("--out", default=MODEL_DIR)
    ap.add_argument(
        "--synthetic", action="store_true", help="합성 데이터면 표시 (성능 증빙에 쓰지 않음)"
    )
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = load_diesel_rows(args.csv)
    frame = DailyFrame(rows)
    answered = [i for i in range(len(rows)) if frame.targets(i) is not None]
    val_idx = answered[-args.holdout_days :]
    train_idx = [i for i in answered if i <= val_idx[0] - 7 * HORIZONS - 1]

    Xtr, Ytr, kept_tr = build_windows(frame, train_idx)
    Xva, _, kept_va = build_windows(frame, val_idx)
    scaler = FeatureScaler().fit(Xtr, Ytr)
    Xtr_s = np.array([scaler.transform(s) for s in Xtr], dtype="float32")
    Ytr_s = np.array([scaler.scale_y(y) for y in Ytr], dtype="float32")
    Xva_s = np.array([scaler.transform(s) for s in Xva], dtype="float32")
    recent = frame.dates[kept_tr[-1]].toordinal() - 365
    weights = np.array(
        [3.0 if frame.dates[i].toordinal() >= recent else 1.0 for i in kept_tr], dtype="float32"
    )

    models, epochs = [], []
    for seed in seeds:
        m, n = train(Xtr_s, Ytr_s, weights, seed)
        models.append(m), epochs.append(n)
        print(f"seed {seed}: {n} epoch")
    forecaster = DieselForecaster(models, scaler)

    changes = forecaster.predict_changes(Xva_s)
    pred = np.array([frame.apply_policy(i, list(c)) for i, c in zip(kept_va, changes)])
    true = np.array([frame.actual(i) for i in kept_va])
    naive = np.array([[frame.price[i]] * HORIZONS for i in kept_va])
    model_rmse = [rmse(true[:, k] - pred[:, k]) for k in range(HORIZONS)]
    naive_rmse = [rmse(true[:, k] - naive[:, k]) for k in range(HORIZONS)]

    d = frame.dates
    meta = {
        "data": args.csv,
        "synthetic": args.synthetic,
        "data_period": [d[0].isoformat(), d[-1].isoformat()],
        "rows": len(rows),
        "train_target_period": [d[kept_tr[0]].isoformat(), d[kept_tr[-1]].isoformat()],
        "validation_period": [d[kept_va[0]].isoformat(), d[kept_va[-1]].isoformat()],
        "train_windows": len(kept_tr),
        "validation_windows": len(kept_va),
        "seeds": seeds,
        "epochs": epochs,
        "rmse": model_rmse,
        "naive_rmse": naive_rmse,
    }
    print(f"데이터 {meta['data_period']} {len(rows)}행 ({'합성' if args.synthetic else '실측'})")
    print(
        f"학습 타깃 {meta['train_target_period']} {len(kept_tr)}개 · 검증 {meta['validation_period']} {len(kept_va)}개"
    )
    print(f"{'k주차 평균 RMSE 원/L':22}" + "".join(f"{k + 1}주".rjust(9) for k in range(HORIZONS)))
    print(f"{'naive (마지막 입력일 가격)':22}" + "".join(f"{v:9.1f}" for v in naive_rmse))
    print(f"{'LSTM + 정책 규칙':22}" + "".join(f"{v:9.1f}" for v in model_rmse))
    print(
        "naive보다 낮음: "
        + ", ".join(
            f"{k + 1}주 {'예' if m < n else '아니오'}"
            for k, (m, n) in enumerate(zip(model_rmse, naive_rmse))
        )
    )
    forecaster.save(args.out, meta)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
