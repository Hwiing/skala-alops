"""경유 데이터 담당의 입력 계약. 공통 서비스 전환 전에는 휘발유 경로와 분리한다."""

from datetime import date, timedelta
from math import isfinite

DIESEL_FEATURE_COLUMNS = (
    "diesel_price",
    "singapore_diesel_price",
    "usd_krw",
    "tax_or_supply_feature",
)
SELECTED_DIESEL_DATASET = "data/processed/diesel_features_2008_spliced.csv"
WEEKLY_HORIZON_DAYS = 7
WEEKLY_HORIZONS = 4


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


def build_diesel_weekly_sequences(
    rows: list[dict], seq_len: int
) -> tuple[list[list[dict]], list[list[float]]]:
    """과거 seq_len일을 입력으로, 다음 1~4주 평균 경유 가격을 정답으로 만든다.

    첫 주는 마지막 입력일 다음날부터 7일이다. 입력 길이는 모델 담당이 확정할 때
    전달하므로 이 함수에서 고정하지 않는다.
    """
    if seq_len < 1:
        raise ValueError("seq_len은 1 이상이어야 합니다")
    checked = validate_diesel_rows(rows)
    horizon = WEEKLY_HORIZON_DAYS * WEEKLY_HORIZONS
    if len(checked) < seq_len + horizon:
        raise ValueError(f"입력 {seq_len}일과 정답 {horizon}일이 필요합니다")
    inputs, targets = [], []
    for start in range(len(checked) - seq_len - horizon + 1):
        end = start + seq_len
        inputs.append(checked[start:end])
        targets.append(
            [
                sum(
                    row["diesel_price"]
                    for row in checked[
                        end + week * WEEKLY_HORIZON_DAYS : end + (week + 1) * WEEKLY_HORIZON_DAYS
                    ]
                )
                / WEEKLY_HORIZON_DAYS
                for week in range(WEEKLY_HORIZONS)
            ]
        )
    return inputs, targets
