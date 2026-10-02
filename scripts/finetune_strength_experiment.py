"""fine-tuning 강도 실험 (docs/evidence/11). 실행: python scripts/finetune_strength_experiment.py

Production(검증 365일 전까지 학습)을 고정하고, 5개 시점의 최근 627행으로
① warm start 강도 6가지 ② 처음부터 재학습을 해 같은 90일 검증에서 Production과 비교한다.
설정마다 1~4주 RMSE와 실제 배포 게이트(check_gate) 통과 여부를 기록해 OUT(JSON)에 저장한다.
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
from serving_app.diesel_gate import check_gate  # noqa: E402
from serving_app.diesel_model import DieselForecaster  # noqa: E402
from serving_app.diesel_training import evaluate, finetune, fit, split_holdout  # noqa: E402

CSV = "data/processed/diesel_features_2008_spliced.csv"
SEEDS = [42, 7, 2026]
ORIGINS = ["2026-01-31", "2026-03-31", "2026-05-31", "2026-07-31", "2026-09-30"]
OUT = "docs/evidence/11_fine_tuning_강도_실험.json"
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


def scored(ev: dict, base: dict) -> dict:
    """1~4주 RMSE와 같은 검증 구간 Production 기준 게이트 판정."""
    gate = check_gate(ev["rmse"], ev["naive_rmse"], base["rmse"])
    return {
        "rmse": [round(v, 2) for v in ev["rmse"]],
        "mean": round(mean(ev["rmse"]), 2),
        "gate": gate["passed"],
        "reasons": gate["reasons"],
    }


def main():
    rows = load_diesel_rows(CSV)
    full = DailyFrame(rows)
    prod, _ = fit(full, split_holdout(full, 365)[0], SEEDS)
    results = []
    for origin in ORIGINS:
        upto = [r for r in rows if r["date"] <= origin]
        f = DailyFrame(upto[-FINETUNE_MIN_ROWS:])
        train_idx, val_idx = split_finetune(f)
        base = evaluate(prod, f, val_idx)
        out = {
            "origin": origin,
            "validation": base["validation_period"],
            "naive": [round(v, 2) for v in base["naive_rmse"]],
            "production": [round(v, 2) for v in base["rmse"]],
            "runs": {},
        }
        for lr, epochs in CONFIGS:
            keras.utils.set_random_seed(0)
            fc = clone(prod)
            finetune(fc, f, train_idx, epochs, lr)
            out["runs"][f"warm {lr:g}x{epochs}"] = scored(evaluate(fc, f, val_idx), base)
        # 처음부터 재학습: 그 시점까지 전체 데이터, 검증 90일과 겹치지 않게.
        # 검증 구간이 같으므로(마지막 90일) Production 비교도 같은 날짜로 한다
        g = DailyFrame(upto)
        _, gval = split_finetune(g)
        gtrain = [
            i
            for i in range(len(g.dates))
            if g.targets(i) is not None and i <= gval[0] - 7 * HORIZONS - 1
        ]
        ev = evaluate(fit(g, gtrain, SEEDS)[0], g, gval)
        out["runs"]["scratch"] = scored(ev, evaluate(prod, g, gval))
        print(json.dumps(out, ensure_ascii=False), flush=True)
        results.append(out)
    with open(OUT, "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
