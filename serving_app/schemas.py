"""경유 v2 요청·응답·모델/AIOps 결과 타입. 공개 JSON 계약은 docs/contracts.md 참조."""

from datetime import date, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from data.contracts import (
    BATCH_MIN_ROWS as BATCH_MIN_ROWS,
)
from data.contracts import (
    CONTRACT_VERSION,
    HORIZON_DAYS,
    HORIZONS,
    INPUT_DAYS,
    parse_iso_day,
)
from data.contracts import (
    PAIR_WINDOW as PAIR_WINDOW,
)

IsoDate = Annotated[date, BeforeValidator(parse_iso_day)]
Price = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Rmse = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PriceVector = Annotated[list[Price], Field(min_length=HORIZONS, max_length=HORIZONS)]
RmseVector = Annotated[list[Rmse], Field(min_length=HORIZONS, max_length=HORIZONS)]


def _registry_version(value):
    """MLflow의 정수 버전은 공개 계약의 문자열로 정규화한다."""
    if type(value) is int:
        if value <= 0:
            raise ValueError("레지스트리 버전 번호는 양의 정수여야 합니다")
        return str(value)
    return value


RegistryVersion = Annotated[
    str,
    BeforeValidator(_registry_version, json_schema_input_type=str | Annotated[int, Field(gt=0)]),
]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DailyPoint(ContractModel):
    date: IsoDate
    diesel_price: Price = Field(description="전국 평균 자동차용경유 원/L")
    singapore_diesel_price: Price = Field(description="싱가포르 경유 USD/bbl, D-1까지 관측")
    usd_krw: Price = Field(description="KRW/USD, D일까지 관측")
    tax_or_supply_feature: float = Field(description="경유 기본 탄력세율 대비 인하율 %, 유한값")


class HealthResponse(ContractModel):
    status: Literal["ok"] = "ok"
    contract_version: str = CONTRACT_VERSION
    model_loaded: bool
    model_version: str | None
    model_source: Literal["local", "mlflow"]
    loading_mode: Literal["lazy", "eager"]


def _consecutive(points):
    for prev, cur in zip(points, points[1:]):
        if cur.date != prev.date + timedelta(days=1):
            raise ValueError(
                f"date는 중복·누락 없이 하루 간격이어야 합니다: {prev.date} → {cur.date}"
            )
    return points


class PredictRequest(ContractModel):
    sequence: list[DailyPoint] = Field(min_length=INPUT_DAYS, max_length=INPUT_DAYS)
    _check_dates = field_validator("sequence")(_consecutive)


class PredictionValues(ContractModel):
    """pyfunc 반환값의 경계 검증. 1주차부터 정확히 4개, 양수·유한 원/L."""

    values: PriceVector


class WeekPrediction(ContractModel):
    horizon_week: int = Field(ge=1, le=HORIZONS)
    start_date: IsoDate
    end_date: IsoDate
    predicted_avg_price: Price

    @model_validator(mode="after")
    def seven_days(self):
        if self.end_date != self.start_date + timedelta(days=HORIZON_DAYS - 1):
            raise ValueError("주차 날짜 구간은 정확히 7일이어야 합니다")
        return self


class PredictResponse(ContractModel):
    predictions: list[WeekPrediction] = Field(min_length=HORIZONS, max_length=HORIZONS)
    base_date: IsoDate
    model_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def weeks_match_base(self):
        for k, week in enumerate(self.predictions, start=1):
            start = self.base_date + timedelta(days=HORIZON_DAYS * (k - 1) + 1)
            if week.horizon_week != k or week.start_date != start:
                raise ValueError("예측은 기준일 다음날부터 1~4주 순서여야 합니다")
        return self


class BatchTestRequest(ContractModel):
    rows: list[DailyPoint] = Field(min_length=BATCH_MIN_ROWS)
    _check_dates = field_validator("rows")(_consecutive)


class BatchPair(ContractModel):
    """date=마지막 입력일. actual[k]=그 다음 k주 7일 평균, naive=기준일 가격."""

    date: IsoDate
    predicted: PriceVector
    actual: PriceVector
    naive: Price


class ReloadResult(ContractModel):
    reloaded: bool
    version: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def verified_swap(self):
        if self.reloaded and (not self.version or self.error is not None):
            raise ValueError("교체 성공에는 실제 서빙 버전이 필요하고 error는 없어야 합니다")
        if not self.reloaded and not self.error:
            raise ValueError("교체 실패에는 error가 필요합니다")
        return self


class RetrainMetrics(ContractModel):
    rmse: RmseVector | None = None
    naive_rmse: RmseVector | None = None
    production_rmse: RmseVector | None = None
    production_before: RegistryVersion | None = None
    passed: bool | None = None
    run_id: str | None = None
    reasons: list[str] = Field(default_factory=list)
    promoted: bool = False
    version: RegistryVersion | None = None

    @model_validator(mode="after")
    def promotion_version(self):
        if self.promoted and not self.version:
            raise ValueError("승격 결과에는 레지스트리 version이 필요합니다")
        if self.promoted and self.passed is False:
            raise ValueError("게이트 실패 모델을 승격할 수 없습니다")
        if not self.promoted and self.version is not None:
            raise ValueError("미승격 결과에 새 레지스트리 version을 넣을 수 없습니다")
        return self

    def check_retrain_metrics(self):
        if self.status in ("promoted", "gate_failed"):
            if self.rmse is None or self.naive_rmse is None:
                raise ValueError("재학습 완료에는 4주 rmse와 naive_rmse가 필요합니다")
        if self.status == "no_production" and (
            self.rmse is not None or self.naive_rmse is not None
        ):
            raise ValueError("Production이 없으면 재학습 RMSE는 없습니다")
        return self


class FineTuneResult(RetrainMetrics):
    status: Literal["promoted", "gate_failed", "no_production"]

    @model_validator(mode="after")
    def consistent_status(self):
        if (self.status == "promoted") != self.promoted:
            raise ValueError("status와 promoted가 일치해야 합니다")
        return self.check_retrain_metrics()


class DriftCheck(RetrainMetrics):
    """rmse=재학습 검증 4주, drift_rmse=탐지용 기존 모델 1주. 두 지표를 구분한다."""

    status: Literal[
        "ok", "insufficient_data", "promoted", "gate_failed", "no_production", "retrain_failed"
    ]
    drift_rmse: Rmse | None = None
    drift_naive_rmse: Rmse | None = None
    reload: ReloadResult | None = None

    @model_validator(mode="after")
    def consistent_state(self):
        if (self.status == "promoted") != self.promoted:
            raise ValueError("status와 promoted가 일치해야 합니다")
        if self.reload is not None and not self.promoted:
            raise ValueError("reload 결과는 승격 뒤에만 기록합니다")
        if self.reload is not None and self.reload.reloaded:
            if self.reload.version != f"champion:{self.version}":
                raise ValueError("교체 버전은 실제 승격 버전과 일치해야 합니다")
        return self.check_retrain_metrics()


class BatchTestResponse(ContractModel):
    predictions: list[BatchPair] = Field(min_length=PAIR_WINDOW)
    drift_check: DriftCheck
    _check_dates = field_validator("predictions")(_consecutive)


class EvaluationResponse(ContractModel):
    predictions: list[BatchPair] = Field(min_length=PAIR_WINDOW)
    model_version: str
    _check_dates = field_validator("predictions")(_consecutive)


class TrainingRequest(ContractModel):
    holdout_days: int = Field(default=365, ge=30, le=3650)


class TrainingResult(RetrainMetrics):
    """초기 학습의 게이트·배포 결과. 실제 로드한 버전까지 확인한다."""

    rmse: RmseVector
    naive_rmse: RmseVector
    passed: bool
    run_id: str
    model_uri: str
    reload: ReloadResult | None = None

    @model_validator(mode="after")
    def verified_deployment(self):
        if self.reload is not None and not self.promoted:
            raise ValueError("서빙 교체 결과는 승격 뒤에만 기록합니다")
        if self.reload and self.reload.reloaded:
            if self.reload.version != f"champion:{self.version}":
                raise ValueError("서빙 버전은 승격 버전과 일치해야 합니다")
        return self


class TrainingJob(ContractModel):
    state: Literal["idle", "running", "completed", "failed"] = "idle"
    job_id: str | None = None
    filename: str | None = None
    rows: int | None = None
    holdout_days: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: TrainingResult | None = None
    error: str | None = None
