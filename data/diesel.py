"""경유 데이터의 임시 계약. 팀 공통 계약 전환 전에는 기존 휘발유 경로와 분리한다."""

from datetime import date, timedelta
from math import isfinite

DIESEL_FEATURE_COLUMNS = (
    "diesel_price",
    "singapore_diesel_price",
    "usd_krw",
    "tax_or_supply_feature",
)


def validate_diesel_rows(rows: list[dict]) -> list[dict]:
    """date + 경유 4피처를 하루 간격·양수·유한값으로 검증한다."""
    result = []
    previous = None
    for row in rows:
        missing = [key for key in ("date", *DIESEL_FEATURE_COLUMNS) if row.get(key) in (None, "")]
        if missing:
            raise ValueError(f"필수 값이 비어 있습니다: {missing} (행: {row.get('date')})")
        day = date.fromisoformat(str(row["date"]))
        if previous is not None and day != previous + timedelta(days=1):
            raise ValueError("date는 중복/누락 없이 하루 간격 오름차순이어야 합니다")
        point = {key: float(row[key]) for key in DIESEL_FEATURE_COLUMNS}
        if not all(isfinite(value) for value in point.values()):
            raise ValueError("피처에 NaN/Infinity를 사용할 수 없습니다")
        if any(point[key] <= 0 for key in DIESEL_FEATURE_COLUMNS[:3]):
            raise ValueError("경유 가격·싱가포르 경유 가격·환율은 양수여야 합니다")
        result.append({"date": day.isoformat(), **point})
        previous = day
    if not result:
        raise ValueError("데이터 행이 없습니다")
    return result
