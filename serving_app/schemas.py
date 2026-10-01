"""기존 요청/응답 구조를 휘발유 도메인으로 치환. docs/contracts.md 참조."""

from pydantic import BaseModel, ConfigDict, Field

from data.features import SEQ_LEN


class DailyPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    diesel_price: float = Field(gt=0, description="전국 평균 보통휘발유 원/L")
    singapore_diesel_price: float = Field(gt=0, description="국제유가 USD/barrel")
    usd_krw: float = Field(gt=0, description="KRW/USD")
    tax_or_supply_feature: float = Field(description="정의 확정 전 합성 예제는 0")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence: list[DailyPoint] = Field(min_length=SEQ_LEN, max_length=SEQ_LEN)


class PredictResponse(BaseModel):
    predicted_price: float
    model_version: str


class BatchTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[DailyPoint] = Field(min_length=SEQ_LEN + 1)


class BatchTestResponse(BaseModel):
    predictions: list[float]
    drift_check: dict
