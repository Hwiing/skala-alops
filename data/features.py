"""
휘발유 데이터를 LSTM 입력용 시퀀스로 변환하는 공용 유틸리티.

Day1 baseline 학습(scripts/train_baseline_v1.py), Day2 MLflow 학습
(serving_app/train_and_register.py), Day3 fine-tuning 재학습
(monitoring/retrain_trigger.py)이 모두 이 모듈을 재사용합니다. 시퀀스 정의를
한 곳에서만 관리해야 "서빙 시점 입력"과 "학습 시점 입력"이 어긋나는 실무 사고를
방지할 수 있습니다.

입력 시퀀스: 최근 SEQ_LEN(20)일의 4개 피처
타깃: 그다음 날의 diesel_price
"""

import csv
import pickle

SEQ_LEN = 20
FEATURE_COLUMNS = ("diesel_price", "singapore_diesel_price", "usd_krw", "tax_or_supply_feature")
# 기존 휘발유·두바이유 경로(v1) 전용 별도 계약. 경유 업로드·학습에 쓰면 diesel_price 누락으로 거부된다.
LEGACY_GASOLINE_COLUMNS = (
    "gasoline_price",
    "dubai_crude_price",
    "usd_krw",
    "tax_or_supply_feature",
)


def validate_rows(rows: list[dict], columns: tuple[str, ...] = FEATURE_COLUMNS) -> list[dict]:
    """정규화 CSV 계약(기본 = 경유 공통 계약). 기존 휘발유 경로는 LEGACY_GASOLINE_COLUMNS로 따로 검사한다."""
    from datetime import date, timedelta
    from math import isfinite

    result = []
    previous = None
    for row in rows:
        missing = [key for key in ("date", *columns) if row.get(key) in (None, "")]
        if missing:
            raise ValueError(f"필수 값이 비어 있습니다: {missing} (행: {row.get('date')})")
        day = date.fromisoformat(row["date"])
        if previous is not None and day != previous + timedelta(days=1):
            raise ValueError("date는 중복/누락 없이 하루 간격 오름차순이어야 합니다")
        point = {key: float(row[key]) for key in columns}
        if not all(isfinite(value) for value in point.values()):
            raise ValueError("피처에 NaN/Infinity를 사용할 수 없습니다")
        if any(point[key] <= 0 for key in columns[:3]):
            raise ValueError("가격과 환율은 양수여야 합니다")
        result.append({"date": day.isoformat(), **point})
        previous = day
    return result


def load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, encoding="utf-8-sig") as f:
        return validate_rows(list(csv.DictReader(f)))


class GasolineScaler:
    """원본 min-max 방식 유지. 학습 구간에만 fit 후 Day2/3 및 서빙에서 재사용."""

    def __init__(self):
        self.minimums = {}
        self.maximums = {}

    def fit(self, rows: list[dict]) -> "GasolineScaler":
        for key in FEATURE_COLUMNS:
            values = [row[key] for row in rows]
            self.minimums[key], self.maximums[key] = min(values), max(values)
        return self

    def _scale(self, key: str, value: float) -> float:
        lo, hi = self.minimums[key], self.maximums[key]
        # 상수 피처도 변화가 발생하면 신호가 사라지지 않도록 범위를 1로 둔다.
        return (value - lo) / (hi - lo if hi != lo else 1.0)

    def transform_point(self, point: dict) -> list[float]:
        return [self._scale(key, point[key]) for key in FEATURE_COLUMNS]

    def scale_price(self, price: float) -> float:
        return self._scale("diesel_price", price)

    def inverse_price(self, scaled_price: float) -> float:
        key = "diesel_price"
        lo, hi = self.minimums[key], self.maximums[key]
        return scaled_price * (hi - lo if hi != lo else 1.0) + lo

    def save(self, path: str = "serving_app/models/scaler.pkl"):
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str = "serving_app/models/scaler.pkl") -> "GasolineScaler":
        scaler = cls()
        with open(path, "rb") as f:
            scaler.__dict__.update(pickle.load(f))
        return scaler


def build_sequences(rows: list[dict], scaler: GasolineScaler, seq_len: int = SEQ_LEN):
    """X=(N,20,4), y=다음날 원/L 실값. 시간 순서 유지."""
    points = [scaler.transform_point(row) for row in rows]
    X, y = [], []
    for i in range(len(rows) - seq_len):
        X.append(points[i : i + seq_len])
        y.append(rows[i + seq_len]["diesel_price"])
    return X, y


def train_test_split(X: list, y: list, test_ratio: float = 0.2):
    split_idx = int(len(X) * (1 - test_ratio))
    if not 0 < split_idx < len(X):
        raise ValueError("학습/검증 시퀀스가 모두 필요합니다")
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def scaler_fit_rows(rows: list[dict], test_ratio: float = 0.2, seq_len: int = SEQ_LEN):
    """train_test_split의 학습 시퀀스가 쓰는 행(입력 + 타깃)만 반환한다.

    검증 구간의 타깃 가격이 scaler의 min/max에 섞이면 미래 정보가 학습에 새므로,
    scaler는 이 행들로만 fit한다. (scripts/train_baseline_v1.py와 같은 경계)
    """
    split_idx = int((len(rows) - seq_len) * (1 - test_ratio))
    if not 0 < split_idx < len(rows) - seq_len:
        raise ValueError("학습/검증 시퀀스가 모두 필요합니다")
    return rows[: seq_len + split_idx]


def split_recent_for_finetune(
    rows: list[dict], train_days: int = 30, val_days: int = 7, seq_len: int = SEQ_LEN
):
    """최근 데이터 fine-tuning용 (학습 행, 검증 행) 제안안.

    - 검증: 가장 최근 val_days일을 타깃으로 하는 행 (선행 seq_len일 입력 포함)
    - 학습: 그 직전 train_days일을 타깃으로 하는 행 (선행 seq_len일 입력 포함)
    두 구간의 입력 문맥은 겹칠 수 있지만 **타깃 날짜는 겹치지 않아** 학습에 쓴 정답으로
    게이트를 재검증하지 않는다. 검증 일수(최소 표본)는 모델 담당이 확정한다.
    """
    if train_days < 1 or val_days < 1:
        raise ValueError("train_days와 val_days는 1 이상이어야 합니다")
    needed = seq_len + train_days + val_days
    if len(rows) < needed:
        raise ValueError(f"fine-tuning에 최소 {needed}행이 필요합니다 (현재 {len(rows)}행)")
    return rows[-needed:-val_days], rows[-(seq_len + val_days) :]
