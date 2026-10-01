"""경유 1~4주 평균가 예측 API 계약(docs/contracts.md v2)."""

from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator

from data.diesel_features import HORIZONS, INPUT_DAYS

PAIR_WINDOW = 7 * HORIZONS  # 드리프트 판정에 쓰는 최근 짝 수(28)
BATCH_MIN_ROWS = (
    INPUT_DAYS + PAIR_WINDOW - 1 + 7 * HORIZONS
)  # 입력 120 + 짝 28 − 1 + 4주 정답 28 = 175


class DailyPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    date: date
    diesel_price: float = Field(gt=0, description="전국 평균 경유 원/L")
    singapore_diesel_price: float = Field(gt=0, description="싱가포르 경유 USD/bbl, D-1")
    usd_krw: float = Field(gt=0, description="KRW/USD")
    tax_or_supply_feature: float


def _consecutive(points: list[DailyPoint]) -> list[DailyPoint]:
    for prev, cur in zip(points, points[1:]):
        if cur.date != prev.date + timedelta(days=1):
            raise ValueError(
                f"date는 중복·누락 없이 하루 간격이어야 합니다: {prev.date} → {cur.date}"
            )
    return points


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence: list[DailyPoint] = Field(min_length=INPUT_DAYS, max_length=INPUT_DAYS)
    _check_dates = field_validator("sequence")(_consecutive)


class WeekPrediction(BaseModel):
    horizon_week: int
    start_date: date
    end_date: date
    predicted_avg_price: float


class PredictResponse(BaseModel):
    predictions: list[WeekPrediction]
    base_date: date
    model_version: str


class BatchTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[DailyPoint] = Field(min_length=BATCH_MIN_ROWS)
    _check_dates = field_validator("rows")(_consecutive)


class BatchPair(BaseModel):
    """AIOps에 넘기는 짝. date = 기준일(마지막 입력일), naive = 기준일 가격."""

    date: date
    predicted: list[float]
    actual: list[float]
    naive: float


class BatchTestResponse(BaseModel):
    predictions: list[BatchPair]
    drift_check: dict
