"""경유 v2 공통 상수·CSV 검증. MLflow/TensorFlow/Pydantic 없이 모든 계층이 사용한다."""

from datetime import date, timedelta
from math import isfinite

CONTRACT_VERSION = "diesel-weekly-v2"
FEATURE_COLUMNS = ("diesel_price", "singapore_diesel_price", "usd_krw", "tax_or_supply_feature")
CSV_COLUMNS = ("date", *FEATURE_COLUMNS)
INPUT_DAYS = 120
MODEL_SEQ_LEN = 28
GAP_WINDOW = 90
HORIZONS = 4
HORIZON_DAYS = 7
TARGET_DAYS = HORIZON_DAYS * HORIZONS
PAIR_WINDOW = 28
BATCH_MIN_ROWS = INPUT_DAYS + PAIR_WINDOW - 1 + TARGET_DAYS
FINETUNE_TRAIN_DAYS = 365
FINETUNE_VAL_DAYS = 90
FINETUNE_MIN_ROWS = (
    GAP_WINDOW
    + MODEL_SEQ_LEN
    - 2
    + FINETUNE_TRAIN_DAYS
    + TARGET_DAYS
    + FINETUNE_VAL_DAYS
    + TARGET_DAYS
)
MODEL_NAME = "DieselPricePredictor"
MODEL_ALIAS = "champion"
LOCAL_PYFUNC_PATH = "serving_app/models/diesel_pyfunc"
WEEK1_RMSE_MAX = 50.0


def parse_iso_day(value) -> date:
    """YYYY-MM-DD 또는 내부 date만 허용한다. epoch·compact 날짜·datetime은 거부한다."""
    if type(value) is date:
        return value
    if not isinstance(value, str):
        raise ValueError("date는 YYYY-MM-DD 문자열이어야 합니다")
    day = date.fromisoformat(value)
    if day.isoformat() != value:
        raise ValueError("date는 YYYY-MM-DD 문자열이어야 합니다")
    return day


def validate_daily_rows(rows: list[dict]) -> list[dict]:
    """공통 CSV 계약: 연속 날짜·양수 가격/환율·유한 피처. 추가 CSV 열은 입력에서 제외한다."""
    result = []
    previous = None
    for row in rows:
        missing = [key for key in CSV_COLUMNS if row.get(key) in (None, "")]
        if missing:
            raise ValueError(f"필수 값이 비어 있습니다: {missing} (행: {row.get('date')})")
        day = parse_iso_day(row["date"])
        if previous is not None and day != previous + timedelta(days=1):
            raise ValueError("date는 중복/누락 없이 하루 간격 오름차순이어야 합니다")
        point = {key: float(row[key]) for key in FEATURE_COLUMNS}
        if not all(isfinite(value) for value in point.values()):
            raise ValueError("피처에 NaN/Infinity를 사용할 수 없습니다")
        if any(point[key] <= 0 for key in FEATURE_COLUMNS[:3]):
            raise ValueError("가격과 환율은 양수여야 합니다")
        result.append({"date": day.isoformat(), **point})
        previous = day
    if not result:
        raise ValueError("데이터 행이 없습니다")
    return result
