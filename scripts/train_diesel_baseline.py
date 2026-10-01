"""
#9 경유 LSTM baseline 학습 + naive 비교 (contracts.md v2). MLflow 없이 로컬 파일로 저장한다.
학습·평가 규칙은 serving_app/diesel_training.py (시간순 holdout, 학습 구간 scaler, 최근 1년 가중).

출력: 주차별 RMSE 비교표, serving_app/models/diesel/ (seed별 모델, scaler, meta.json)
실행: python scripts/train_diesel_baseline.py [--csv data/processed/diesel_features_2008_spliced.csv]
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")  # 작은 LSTM은 단일 스레드가 더 빠름
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.diesel_features import DailyFrame, load_diesel_rows  # noqa: E402
from serving_app.diesel_model import MODEL_DIR  # noqa: E402
from serving_app.diesel_training import evaluate, fit, report, split_holdout  # noqa: E402

DEFAULT_CSV = "data/processed/diesel_features_2008_spliced.csv"  # 데이터 담당 PR #32 산출물


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--holdout-days", type=int, default=365)
    ap.add_argument("--seeds", default="42,7,2026")
    ap.add_argument("--out", default=MODEL_DIR)
    ap.add_argument(
        "--synthetic", action="store_true", help="합성 데이터면 표시 (성능 증빙에 쓰지 않음)"
    )
    args = ap.parse_args()

    rows = load_diesel_rows(args.csv)
    frame = DailyFrame(rows)
    train_idx, val_idx = split_holdout(frame, args.holdout_days)
    forecaster, info = fit(frame, train_idx, [int(s) for s in args.seeds.split(",")])
    meta = {
        "data": args.csv,
        "synthetic": args.synthetic,
        "data_period": [frame.dates[0].isoformat(), frame.dates[-1].isoformat()],
        "rows": len(rows),
        **info,
        **evaluate(forecaster, frame, val_idx),
    }
    print(report(meta))
    beats = [m < n for m, n in zip(meta["rmse"], meta["naive_rmse"])]
    print(
        "naive보다 낮음: "
        + ", ".join(f"{k + 1}주 {'예' if b else '아니오'}" for k, b in enumerate(beats))
    )
    forecaster.save(args.out, meta)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
