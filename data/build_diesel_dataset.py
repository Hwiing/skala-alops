"""경유·싱가포르 경유·환율 실측 원본과 경유 유류세 구간표로 4피처 CSV를 만든다.

공통 학습·서빙 계약은 아직 gasoline_price이므로 이 결과를 기존 API에 업로드하지 않는다.
"""

import csv
from datetime import date

from data.diesel import DIESEL_FEATURE_COLUMNS, validate_diesel_rows
from data.external import (
    FX_LAG_DAYS,
    MAX_FX_STALE_DAYS,
    SINGAPORE_DIESEL_LAG_DAYS,
    asof_values,
    first_available_day,
    load_opinet_singapore_diesel,
    load_tax_policy,
    load_usd_krw,
    tax_cut_rates,
)
from data.opinet import load_opinet_diesel_many

DIESEL_TAX_POLICY_PATH = "data/reference/diesel_fuel_tax_cut.csv"
DEFAULT_OUT = "data/processed/diesel_features.csv"


def build_rows(diesel: list[dict], singapore, fx, tax_periods) -> tuple[list[dict], int]:
    start = max(
        first_available_day(singapore, SINGAPORE_DIESEL_LAG_DAYS),
        first_available_day(fx, FX_LAG_DAYS),
    )
    kept = [row for row in diesel if date.fromisoformat(row["date"]) >= start]
    days = [date.fromisoformat(row["date"]) for row in kept]
    columns = {
        "singapore_diesel_price": asof_values(days, singapore, SINGAPORE_DIESEL_LAG_DAYS),
        "usd_krw": asof_values(days, fx, FX_LAG_DAYS, max_stale_days=MAX_FX_STALE_DAYS),
        "tax_or_supply_feature": tax_cut_rates(days, tax_periods),
    }
    rows = [
        {**row, **{name: values[i] for name, values in columns.items()}}
        for i, row in enumerate(kept)
    ]
    return validate_diesel_rows(rows), len(diesel) - len(kept)


def write_rows(rows: list[dict], path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", *DIESEL_FEATURE_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="실측 경유 4피처 정규화 CSV 생성")
    parser.add_argument(
        "--diesel",
        nargs="+",
        default=["data/raw/opinet_diesel_2012_2023.csv", "data/raw/opinet_diesel.csv"],
    )
    parser.add_argument(
        "--singapore",
        nargs="+",
        default=[
            "data/raw/opinet_singapore_diesel_2012_2023.csv",
            "data/raw/opinet_singapore_diesel_2023_2026.csv",
        ],
    )
    parser.add_argument("--fx", default="data/raw/ecos_usdkrw_2012_2026.csv")
    parser.add_argument("--tax", default=DIESEL_TAX_POLICY_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    diesel = load_opinet_diesel_many(args.diesel)
    rows, trimmed = build_rows(
        diesel,
        load_opinet_singapore_diesel(args.singapore),
        load_usd_krw(args.fx),
        load_tax_policy(args.tax),
    )
    write_rows(rows, args.out)
    print(f"경유 {len(diesel)}행 -> 앞쪽 {trimmed}행 제외 -> {len(rows)}행")
    print(f"기간 {rows[0]['date']} ~ {rows[-1]['date']} -> {args.out}")


if __name__ == "__main__":
    main()
