"""
원본 3종 + 유류세 구간표 -> 학습·업로드용 정규화 CSV (date + FEATURE_COLUMNS).

    .venv/bin/python -m data.build_dataset \\
        --gasoline data/raw/opinet_gasoline.csv \\
        --crude data/raw/opinet_crude.csv \\
        --fx data/raw/ecos_usdkrw.csv

국제유가·환율 원본은 휘발유 시작일보다 2주 정도 앞에서부터 받아 두세요. 앞쪽에
공개된 외부 값이 없는 날짜는 추정하지 않고 잘라냅니다(잘린 행 수를 출력).
"""

import csv
from datetime import date

from data.external import (
    CRUDE_LAG_DAYS,
    FX_LAG_DAYS,
    TAX_POLICY_PATH,
    asof_values,
    first_available_day,
    load_opinet_crude,
    load_tax_policy,
    load_usd_krw,
    tax_cut_rates,
)
from data.features import FEATURE_COLUMNS, validate_rows
from data.opinet import load_opinet_gasoline

DEFAULT_OUT = "data/processed/gasoline_features.csv"


def build_rows(gasoline: list[dict], crude, fx, tax_periods) -> tuple[list[dict], int]:
    start = max(first_available_day(crude, CRUDE_LAG_DAYS), first_available_day(fx, FX_LAG_DAYS))
    kept = [row for row in gasoline if date.fromisoformat(row["date"]) >= start]
    days = [date.fromisoformat(row["date"]) for row in kept]

    columns = {
        "crude_oil_price": asof_values(days, crude, CRUDE_LAG_DAYS),
        "usd_krw": asof_values(days, fx, FX_LAG_DAYS),
        "tax_or_supply_feature": tax_cut_rates(days, tax_periods),
    }
    rows = [
        {**row, **{name: values[i] for name, values in columns.items()}}
        for i, row in enumerate(kept)
    ]
    # 공통 계약(하루 간격·양수·유한값)을 업로드 라우터와 같은 함수로 한 번 더 확인한다.
    validated = validate_rows([{k: str(v) for k, v in row.items()} for row in rows])
    return validated, len(gasoline) - len(kept)


def write_rows(rows: list[dict], path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", *FEATURE_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="휘발유 4피처 정규화 CSV 생성")
    parser.add_argument("--gasoline", default="data/raw/opinet_gasoline.csv")
    parser.add_argument("--crude", default="data/raw/opinet_crude.csv")
    parser.add_argument("--fx", default="data/raw/ecos_usdkrw.csv")
    parser.add_argument("--tax", default=TAX_POLICY_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    gasoline = load_opinet_gasoline(args.gasoline)
    rows, trimmed = build_rows(
        gasoline, load_opinet_crude(args.crude), load_usd_krw(args.fx), load_tax_policy(args.tax)
    )
    write_rows(rows, args.out)
    print(f"휘발유 {len(gasoline)}행 -> 앞쪽 {trimmed}행 제외(외부 값 공개 전) -> {len(rows)}행")
    print(f"기간 {rows[0]['date']} ~ {rows[-1]['date']} -> {args.out}")


if __name__ == "__main__":
    main()
