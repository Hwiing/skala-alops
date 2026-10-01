"""2008년부터 경유 후보를 생성한다. 2012년 이전 국제가격은 규격 접합 추정값이다."""

import csv
from bisect import bisect_right
from datetime import date, timedelta
from statistics import mean

from data.build_diesel_dataset import build_rows, write_rows
from data.diesel import SELECTED_DIESEL_DATASET
from data.external import (
    SINGAPORE_DIESEL_LAG_DAYS,
    load_ecos_wide_usd_krw,
    load_opinet_singapore_diesel,
    load_tax_policy,
    load_usd_krw,
    merge_observed_series,
)
from data.opinet import load_opinet_diesel_many

SINGAPORE_005_COLUMN = "경유(0.05%)"
OVERLAP_DAYS = 60
DEFAULT_OUT = SELECTED_DIESEL_DATASET
DEFAULT_PROVENANCE = "data/processed/diesel_features_2008_provenance.csv"


def calculate_spread(overlap_path: str) -> float:
    low = dict(load_opinet_singapore_diesel([overlap_path], column=SINGAPORE_005_COLUMN))
    ultra_low = load_opinet_singapore_diesel([overlap_path])
    paired = [(day, high - low[day]) for day, high in ultra_low if day in low]
    if len(paired) < OVERLAP_DAYS:
        raise ValueError(f"규격 겹침 거래일이 {OVERLAP_DAYS}일보다 적습니다")
    first = paired[:OVERLAP_DAYS]
    if first[0][0] != date(2012, 12, 3):
        raise ValueError("규격 겹침 첫 관측일이 2012-12-03이 아닙니다")
    return mean(delta for _, delta in first)


def splice_singapore(
    low_grade: list[tuple[date, float]],
    ultra_low_grade: list[tuple[date, float]],
    spread: float,
) -> tuple[list[tuple[date, float]], date]:
    """0.001% 첫 관측 전까지만 0.05% + 스프레드를 쓴다."""
    if not low_grade or not ultra_low_grade:
        raise ValueError("두 규격의 관측이 모두 필요합니다")
    transition = ultra_low_grade[0][0]
    estimated = [(day, price + spread) for day, price in low_grade if day < transition]
    if not estimated or estimated[-1][0] + timedelta(days=1) > transition:
        raise ValueError("접합 전 구간을 확인할 수 없습니다")
    if any(price <= 0 or price > 500 for _, price in estimated):
        raise ValueError("접합 가격이 USD/bbl 범위 밖입니다")
    return estimated + ultra_low_grade, transition


def provenance_rows(rows: list[dict], singapore: list[tuple[date, float]], transition: date):
    observations = [day for day, _ in singapore]
    result = []
    for row in rows:
        cutoff = date.fromisoformat(row["date"]) - timedelta(days=SINGAPORE_DIESEL_LAG_DAYS)
        observation = observations[bisect_right(observations, cutoff) - 1]
        result.append(
            {
                "date": row["date"],
                "singapore_diesel_spliced": observation < transition,
                "singapore_price_observation_date": observation.isoformat(),
            }
        )
    return result


def write_provenance(rows: list[dict], path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "singapore_diesel_spliced", "singapore_price_observation_date"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="2008년부터 경유 후보와 접합 출처 생성")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--provenance", default=DEFAULT_PROVENANCE)
    args = parser.parse_args()

    diesel = load_opinet_diesel_many(
        [
            "data/raw/opinet_diesel_2008_2012.csv",
            "data/raw/opinet_diesel_2012_2023.csv",
            "data/raw/opinet_diesel.csv",
        ]
    )
    low = load_opinet_singapore_diesel(
        ["data/raw/opinet_singapore_diesel_2008_2012.csv"], column=SINGAPORE_005_COLUMN
    )
    high = load_opinet_singapore_diesel(
        [
            "data/raw/opinet_singapore_diesel_2012_2023.csv",
            "data/raw/opinet_singapore_diesel_2023_2026.csv",
        ]
    )
    spread = calculate_spread("data/raw/opinet_singapore_diesel_grade_overlap_2012_2013.csv")
    singapore, transition = splice_singapore(low, high, spread)
    fx = merge_observed_series(
        load_ecos_wide_usd_krw("data/raw/ecos_usdkrw_wide_2008_2012.csv"),
        load_usd_krw("data/raw/ecos_usdkrw_2012_2026.csv"),
    )
    rows, trimmed = build_rows(
        diesel, singapore, fx, load_tax_policy("data/reference/diesel_fuel_tax_cut.csv")
    )
    write_rows(rows, args.out)
    write_provenance(provenance_rows(rows, singapore, transition), args.provenance)
    print(f"경유 {len(diesel)}행 -> 앞쪽 {trimmed}행 제외 -> {len(rows)}행")
    print(f"기간 {rows[0]['date']} ~ {rows[-1]['date']}, 접합 스프레드 {spread:.6f} USD/bbl")
    print(f"4피처: {args.out}\n출처 플래그: {args.provenance}")


if __name__ == "__main__":
    main()
