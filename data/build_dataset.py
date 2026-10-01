"""
[기존 휘발유 v1 경로] 원본 3종 + 유류세 구간표 -> 휘발유 정규화 CSV (date + LEGACY_GASOLINE_COLUMNS).

경유 학습·업로드(계약 v2)에는 쓰지 않는다: 출력 컬럼이 gasoline_price·dubai_crude_price라
경유 공통 계약 검사(diesel_price 필수)에서 거부된다. 경유는 data.build_diesel_dataset을 쓴다.

    .venv/bin/python -m data.build_dataset \\
        --gasoline data/raw/opinet_gasoline.csv \\
        --crude data/raw/opinet_crude.csv \\
        --fx data/raw/ecos_usdkrw.csv

국제유가·환율 원본은 휘발유 시작일보다 2주 정도 앞에서부터 받아 두세요. 앞쪽에
공개된 외부 값이 없는 날짜는 추정하지 않고 잘라냅니다(잘린 행 수를 출력).
"""

import csv
from datetime import date, timedelta
from math import isfinite

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
from data.opinet import load_opinet_gasoline

DEFAULT_OUT = "data/processed/gasoline_features.csv"
LEGACY_GASOLINE_COLUMNS = (
    "gasoline_price",
    "dubai_crude_price",
    "usd_krw",
    "tax_or_supply_feature",
)


def validate_legacy_gasoline_rows(rows: list[dict]) -> list[dict]:
    """과거 휘발유·Dubai 결과만 검사한다. 경유 공통 계약과 의도적으로 분리한다."""
    result = []
    previous = None
    for row in rows:
        missing = [key for key in ("date", *LEGACY_GASOLINE_COLUMNS) if row.get(key) in (None, "")]
        if missing:
            raise ValueError(f"필수 값이 비어 있습니다: {missing} (행: {row.get('date')})")
        day = date.fromisoformat(row["date"])
        if previous is not None and day != previous + timedelta(days=1):
            raise ValueError("date는 중복/누락 없이 하루 간격 오름차순이어야 합니다")
        point = {key: float(row[key]) for key in LEGACY_GASOLINE_COLUMNS}
        if not all(isfinite(value) for value in point.values()):
            raise ValueError("피처에 NaN/Infinity를 사용할 수 없습니다")
        if any(point[key] <= 0 for key in LEGACY_GASOLINE_COLUMNS[:3]):
            raise ValueError("가격과 환율은 양수여야 합니다")
        result.append({"date": day.isoformat(), **point})
        previous = day
    if not result:
        raise ValueError("데이터 행이 없습니다")
    return result


def build_rows(gasoline: list[dict], crude, fx, tax_periods) -> tuple[list[dict], int]:
    start = max(first_available_day(crude, CRUDE_LAG_DAYS), first_available_day(fx, FX_LAG_DAYS))
    kept = [row for row in gasoline if date.fromisoformat(row["date"]) >= start]
    days = [date.fromisoformat(row["date"]) for row in kept]

    columns = {
        "dubai_crude_price": asof_values(days, crude, CRUDE_LAG_DAYS),
        "usd_krw": asof_values(days, fx, FX_LAG_DAYS),
        "tax_or_supply_feature": tax_cut_rates(days, tax_periods),
    }
    rows = [
        {**row, **{name: values[i] for name, values in columns.items()}}
        for i, row in enumerate(kept)
    ]
    # 하루 간격·양수·유한값을 휘발유 전용 계약으로 검사한다.
    validated = validate_legacy_gasoline_rows([{k: str(v) for k, v in row.items()} for row in rows])
    return validated, len(gasoline) - len(kept)


def write_rows(rows: list[dict], path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", *LEGACY_GASOLINE_COLUMNS])
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
