"""
#15 판정 기준 실측 검증: 과거 1년을 "실시간 운영"처럼 다시 돌려 드리프트 판정이 언제 울리는지 본다.

    1) test_end 기준 마지막 holdout_days일을 시험 구간으로 떼고, 그 이전 데이터로만 LSTM을 학습
       (diesel_training.split_holdout/fit - 서빙 모델과 같은 학습 코드, 미래 정보 누수 없음)
    2) 시험 구간의 기준일마다 batch_test와 같은 짝 {date, predicted[4], actual[4], naive}을 만든다
    3) 기준일 28개씩 밀면서 drift_detector.evaluate()로 판정
       판정 가능일 = 마지막 기준일 + 7일 (1주차 정답이 그때 도착)
    4) 월별 drift 비율과 1주차 모델/naive RMSE를 표로 출력

실행 (TensorFlow 필요, 구간당 1~2분):
    python scripts/backtest_drift.py --csv data/processed/diesel_features_2008_spliced.csv \
        --test-end 2022-09-30 --out logs/backtest_2022.md
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.contracts import INPUT_DAYS, PAIR_WINDOW  # noqa: E402
from data.diesel_features import WINDOWS, DailyFrame, load_diesel_rows  # noqa: E402
from serving_app.monitoring.drift_detector import evaluate  # noqa: E402


def make_pairs(forecaster, rows: list[dict], base_indices: list[int]) -> list[dict]:
    prices = [r["diesel_price"] for r in rows]
    pairs = []
    for b in base_indices:
        if b - INPUT_DAYS + 1 < 0 or b + max(WINDOWS[-1]) >= len(rows):
            continue
        pairs.append(
            {
                "date": rows[b]["date"],
                "predicted": forecaster.predict(rows[b - INPUT_DAYS + 1 : b + 1]),
                "actual": [sum(prices[b + d] for d in w) / 7 for w in WINDOWS],
                "naive": prices[b],
            }
        )
    return pairs


def rolling(pairs: list[dict]) -> list[dict]:
    out = []
    for t in range(PAIR_WINDOW - 1, len(pairs)):
        r = evaluate(pairs[t - PAIR_WINDOW + 1 : t + 1])
        judged_on = date.fromisoformat(pairs[t]["date"]) + timedelta(days=7)
        out.append({**r, "judged_on": judged_on.isoformat()})
    return out


def monthly_table(results: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in results:
        by[r["judged_on"][:7]].append(r)
    table = []
    for month in sorted(by):
        rs = by[month]
        drift = sum(r["status"] == "drift" for r in rs)
        table.append(
            {
                "month": month,
                "days": len(rs),
                "drift_days": drift,
                "model": sum(r["week1_rmse"] for r in rs) / len(rs),
                "naive": sum(r["naive_rmse"] for r in rs) / len(rs),
            }
        )
    return table


def streaks(results: list[dict]) -> list[tuple[str, str, int]]:
    out, start, prev = [], None, None
    for r in results:
        if r["status"] == "drift":
            start = start or r["judged_on"]
            prev = r["judged_on"]
        elif start:
            out.append(
                (start, prev, (date.fromisoformat(prev) - date.fromisoformat(start)).days + 1)
            )
            start = None
    if start:
        out.append((start, prev, (date.fromisoformat(prev) - date.fromisoformat(start)).days + 1))
    return out


def to_markdown(label: str, info: dict, table: list[dict], results: list[dict]) -> str:
    total = len(results)
    drift = sum(r["status"] == "drift" for r in results)
    lines = [
        f"### {label}",
        "",
        f"- 학습 타깃 {info['train_target_period'][0]} ~ {info['train_target_period'][1]} "
        f"(seed {info['seeds']}), 시험 기준일 {info['test_period'][0]} ~ {info['test_period'][1]}",
        f"- 판정 {total}일 중 drift {drift}일 ({drift / total * 100:.0f}%)",
        "",
        "| 판정 월 | 판정일 | drift일 | 1주차 모델 RMSE(평균) | naive RMSE(평균) |",
        "|---|---|---|---|---|",
    ]
    for m in table:
        mark = " ⚠️" if m["drift_days"] else ""
        lines.append(
            f"| {m['month']} | {m['days']} | {m['drift_days']}{mark} | {m['model']:.1f} | {m['naive']:.1f} |"
        )
    s = [x for x in streaks(results) if x[2] >= 7]
    lines += [
        "",
        "drift 연속 구간(7일 이상): " + (", ".join(f"{a}~{b}({n}일)" for a, b, n in s) or "없음"),
    ]
    return "\n".join(lines)


def main():
    from serving_app.diesel_training import fit, split_holdout

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/diesel_features_2008_spliced.csv")
    ap.add_argument("--test-end", help="이 날짜까지만 데이터 사용 (기본: 마지막 날)")
    ap.add_argument("--holdout-days", type=int, default=365)
    ap.add_argument("--seeds", default="42,7,2026")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out")
    ap.add_argument("--json", help="기준일별 판정 결과 저장 (기준 비교용)")
    args = ap.parse_args()

    rows = load_diesel_rows(args.csv)
    if args.test_end:
        rows = [r for r in rows if r["date"] <= args.test_end]
    frame = DailyFrame(rows)
    train_idx, val_idx = split_holdout(frame, args.holdout_days)
    forecaster, info = fit(frame, train_idx, [int(s) for s in args.seeds.split(",")])
    pairs = make_pairs(forecaster, rows, val_idx)
    info["test_period"] = [pairs[0]["date"], pairs[-1]["date"]]
    results = rolling(pairs)
    if args.json:
        import json

        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"info": info, "results": results}, f, ensure_ascii=False)
    md = to_markdown(
        args.label or f"시험 구간 ~{rows[-1]['date']}", info, monthly_table(results), results
    )
    print(md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
