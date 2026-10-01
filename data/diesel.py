"""경유 데이터 담당의 입력 계약. 공통 서비스 전환 전에는 휘발유 경로와 분리한다."""

from data.contracts import (
    FEATURE_COLUMNS as DIESEL_FEATURE_COLUMNS,
)
from data.contracts import (
    HORIZON_DAYS as WEEKLY_HORIZON_DAYS,
)
from data.contracts import (
    HORIZONS as WEEKLY_HORIZONS,
)
from data.contracts import (
    validate_daily_rows,
)

__all__ = [
    "DIESEL_FEATURE_COLUMNS",
    "WEEKLY_HORIZON_DAYS",
    "WEEKLY_HORIZONS",
    "SELECTED_DIESEL_DATASET",
    "validate_diesel_rows",
    "build_diesel_weekly_sequences",
]

SELECTED_DIESEL_DATASET = "data/processed/diesel_features_2008_spliced.csv"


def validate_diesel_rows(rows: list[dict]) -> list[dict]:
    """데이터 생성·업로드·학습이 같은 공통 CSV 검증을 사용한다."""
    return validate_daily_rows(rows)


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
