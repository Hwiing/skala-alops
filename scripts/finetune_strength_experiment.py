"""fine-tuning 강도 실험 (docs/evidence/11). 실행: python scripts/finetune_strength_experiment.py

Production(검증 365일 전까지 학습)을 고정하고, 5개 시점의 최근 627행으로
① warm start 강도 6가지 ② 처음부터 재학습을 해 같은 90일 검증에서 Production과 비교한다.
"""

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow import keras  # noqa: E402

from data.diesel_features import (  # noqa: E402
    FINETUNE_MIN_ROWS,
    HORIZONS,
    DailyFrame,
    load_diesel_rows,
    split_finetune,
)
from serving_app.diesel_model import DieselForecaster  # noqa: E402
from serving_app.diesel_training import evaluate, finetune, fit, split_holdout  # noqa: E402

CSV = "data/processed/diesel_features_2008_spliced.csv"
SEEDS = [42, 7, 2026]
ORIGINS = ["2026-01-31", "2026-03-31", "2026-05-31", "2026-07-31", "2026-09-30"]
CONFIGS = [(1e-4, 10), (3e-4, 10), (3e-4, 30), (1e-3, 10), (1e-3, 30), (3e-3, 10)]


def clone(fc: DieselForecaster) -> DieselForecaster:
    models = []
    for m in fc.models:
        c = keras.models.clone_model(m)
        c.set_weights(m.get_weights())
        models.append(c)
    return DieselForecaster(models, fc.scaler)


def mean(v):
    return sum(v) / len(v)


def main():
    rows = load_diesel_rows(CSV)
    full = DailyFrame(rows)
    prod, _ = fit(full, split_holdout(full, 365)[0], SEEDS)
    for origin in ORIGINS:
        upto = [r for r in rows if r["date"] <= origin]
        f = DailyFrame(upto[-FINETUNE_MIN_ROWS:])
        train_idx, val_idx = split_finetune(f)
        base = evaluate(prod, f, val_idx)
        out = {"origin": origin, "validation": base["validation_period"]}
        out["naive"], out["production"] = mean(base["naive_rmse"]), mean(base["rmse"])
        for lr, epochs in CONFIGS:
            keras.utils.set_random_seed(0)
            fc = clone(prod)
            finetune(fc, f, train_idx, epochs, lr)
            out[f"warm {lr:g}x{epochs}"] = mean(evaluate(fc, f, val_idx)["rmse"])
        # 처음부터 재학습: 그 시점까지 전체 데이터, 검증 90일과 겹치지 않게
        g = DailyFrame(upto)
        _, gval = split_finetune(g)
        gtrain = [
            i
            for i in range(len(g.dates))
            if g.targets(i) is not None and i <= gval[0] - 7 * HORIZONS - 1
        ]
        out["scratch"] = mean(evaluate(fit(g, gtrain, SEEDS)[0], g, gval)["rmse"])
        print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in out.items()}))


if __name__ == "__main__":
    main()
