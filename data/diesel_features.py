"""
경유 주간 예측(contracts.md v2)의 피처·타깃·정책 규칙. 학습·평가·서빙이 모두 이 모듈을 쓴다.

입력: 정규화 CSV 행 `date,diesel_price,singapore_diesel_price,usd_krw,tax_or_supply_feature` (하루 간격)
모델 입력: 마지막 SEQ_LEN(28)일 × 8피처. 90일 평균 피처 때문에 API는 최근 INPUT_DAYS(120)일을 받는다.
모델 출력: 다음 k주(k=1..4) 평균 "세전 등가 가격"의 변화. 세금 경로·최고가격은 apply_policy()로 더한다.

피처는 그날까지 알려진 값만 쓴다(싱가포르 경유가는 데이터 담당이 D-1 as-of로 넣음). 근거: docs/evidence/06.
"""

import csv
from datetime import date, timedelta
from math import isfinite, sqrt
from statistics import median

from data.diesel_policy import CapSchedule, TaxSchedule

SEQ_LEN = 28
GAP_WINDOW = 90
INPUT_DAYS = 120  # SEQ_LEN + GAP_WINDOW - 1 = 117, 여유 3일
HORIZONS = 4
L_PER_BBL = 158.987
IMPORT_LEVY = 16.0  # 석유 수입부과금 원/L
DIESEL_COLUMNS = ("diesel_price", "singapore_diesel_price", "usd_krw", "tax_or_supply_feature")
WINDOWS = [
    range(7 * (k - 1) + 1, 7 * k + 1) for k in range(1, HORIZONS + 1)
]  # k주 평균에 들어가는 날 오프셋
FEATURES = ("dpt", "dc", "gap", "gap90", "pt_w1", "c_w1", "c_w2", "c_w3")


def load_diesel_rows(path: str) -> list[dict]:
    """정규화 CSV를 읽어 하루 간격·유한값을 확인한다. 상세 검증은 데이터 담당 로더(data/diesel.py)를 따른다."""
    with open(path, encoding="utf-8-sig") as f:
        rows = [
            {"date": r["date"], **{k: float(r[k]) for k in DIESEL_COLUMNS}}
            for r in csv.DictReader(f)
        ]
    for prev, cur in zip(rows, rows[1:]):
        if date.fromisoformat(cur["date"]) != date.fromisoformat(prev["date"]) + timedelta(days=1):
            raise ValueError(f"date는 하루 간격이어야 합니다: {prev['date']} → {cur['date']}")
    if not all(isfinite(v) for r in rows for k, v in r.items() if k != "date"):
        raise ValueError("피처에 NaN/Infinity를 사용할 수 없습니다")
    return rows


class DailyFrame:
    """행 목록에서 날짜별 세전 가격·실효 원가·정책 상태를 계산한다."""

    def __init__(
        self, rows: list[dict], taxes: TaxSchedule | None = None, caps: CapSchedule | None = None
    ):
        self.taxes, self.caps = taxes or TaxSchedule(), caps or CapSchedule()
        self.dates = [date.fromisoformat(r["date"]) for r in rows]
        self.price = [r["diesel_price"] for r in rows]
        tax = [self.taxes.tax(d) for d in self.dates]
        self.cap = [self.caps.cap(d) for d in self.dates]
        cost = [
            (r["singapore_diesel_price"] * r["usd_krw"] / L_PER_BBL + t + IMPORT_LEVY) * 1.1
            for r, t in zip(rows, tax)
        ]
        self.binding = [c is not None and k > c for k, c in zip(cost, self.cap)]
        # 최고가격 기간엔 정유사 공급가가 상한에 묶이므로 실효 원가 = min(국제가 기반 원가, 상한)
        self.c = [min(k, c) if c is not None else k for k, c in zip(cost, self.cap)]
        self.c = [v - 1.1 * t for v, t in zip(self.c, tax)]  # 세전 실효 원가
        self.pre_tax = [p - 1.1 * self.taxes.retail_tax(d) for p, d in zip(self.price, self.dates)]
        self.margin = self._margin(cost)

    def _margin(self, cost):
        """최고가격 기간 주유소 마진 = 최근 28일 (소매가 − 상한) 중앙값. 상한 직전엔 (소매가 − 원가) 28일 중앙값."""
        out, pre = [], []
        for i, (p, c, k) in enumerate(zip(self.price, self.cap, cost)):
            if c is None:  # 상한 발표 직후(시행 전)에도 천장을 쓸 수 있게 직전 마진을 둔다
                pre = (pre + [p - k])[-28:]
                out.append(max(median(pre), 0.0))
                continue
            recent = [
                self.price[j] - self.cap[j]
                for j in range(max(0, i - 27), i + 1)
                if self.cap[j] is not None
            ]
            m = median(recent) if len(recent) >= 7 else (median(pre) if pre else None)
            out.append(max(m, 0.0) if m is not None else None)
        return out

    def features(self, i: int) -> list[float] | None:
        """i일의 8피처. 과거가 부족하면 None."""
        if i < max(GAP_WINDOW - 1, 21):
            return None
        pt, c = self.pre_tax, self.c
        gap = [pt[j] - c[j] for j in range(i - GAP_WINDOW + 1, i + 1)]
        return [
            pt[i] - pt[i - 1],
            c[i] - c[i - 1],
            gap[-1],
            gap[-1] - sum(gap) / GAP_WINDOW,
            pt[i] - pt[i - 7],
            c[i] - c[i - 7],
            c[i - 7] - c[i - 14],
            c[i - 14] - c[i - 21],
        ]

    def targets(self, i: int) -> list[float] | None:
        """학습 타깃: 다음 k주 평균 세전 가격 − 오늘 세전 가격. 미래가 부족하면 None."""
        if i + 7 * HORIZONS >= len(self.price):
            return None
        return [sum(self.pre_tax[i + d] for d in w) / 7 - self.pre_tax[i] for w in WINDOWS]

    def actual(self, i: int) -> list[float] | None:
        if i + 7 * HORIZONS >= len(self.price):
            return None
        return [sum(self.price[i + d] for d in w) / 7 for w in WINDOWS]

    def apply_policy(self, i: int, change: list[float]) -> list[float]:
        """모델이 낸 세전 가격 변화 4개 → 정책 규칙을 적용한 k주 평균 소매가 4개."""
        today = self.dates[i]
        out = []
        for k, w in enumerate(WINDOWS):
            days = [today + timedelta(days=d) for d in w]
            if self.binding[i]:  # 상한(세후)에 묶이면 세금 변화 무관 → 현재가 + 발표된 상한 변화
                out.append(self.price[i] + sum(self.caps.retail_change(u, today) for u in days) / 7)
                continue
            tax_path = sum(1.1 * self.taxes.retail_tax(u, today) for u in days) / 7  # 발표된 변경만
            level = self.pre_tax[i] + change[k] + tax_path
            known = [v for v in (self.caps.known_cap(u, today) for u in days) if v is not None]
            if known and self.margin[i] is not None:  # 상한 + 최근 마진을 넘는 예측은 자른다
                level = min(level, sum(known) / len(known) + self.margin[i] + 5)
            out.append(level)
        return out


class FeatureScaler:
    """학습 구간에만 fit. 표준화 후 학습 구간 최소~최대로 잘라 처음 보는 극단값에 과하게 반응하지 않게 한다."""

    def fit(self, X: list[list[list[float]]], Y: list[list[float]]) -> "FeatureScaler":
        cols = list(zip(*[step for seq in X for step in seq]))
        self.mean = [sum(c) / len(c) for c in cols]
        self.std = [
            sqrt(sum((v - m) ** 2 for v in c) / (len(c) - 1)) or 1.0
            for c, m in zip(cols, self.mean)
        ]
        self.lo, self.hi = [min(c) for c in cols], [max(c) for c in cols]
        ys = list(zip(*Y))
        self.y_std = [
            sqrt(sum(v * v for v in y) / len(y) - (sum(y) / len(y)) ** 2) or 1.0 for y in ys
        ]
        return self

    def transform(self, seq: list[list[float]]) -> list[list[float]]:
        return [
            [
                (min(max(v, lo), hi) - m) / s
                for v, m, s, lo, hi in zip(step, self.mean, self.std, self.lo, self.hi)
            ]
            for step in seq
        ]

    def scale_y(self, y: list[float]) -> list[float]:
        return [v / s for v, s in zip(y, self.y_std)]

    def inverse_y(self, y: list[float]) -> list[float]:
        return [v * s for v, s in zip(y, self.y_std)]


def build_windows(frame: DailyFrame, indices) -> tuple[list, list, list]:
    """indices 날짜별 (SEQ_LEN, 8) 입력과 4개 타깃. 과거·미래가 부족한 날은 건너뛴다."""
    feats = {}
    X, Y, kept = [], [], []
    for i in indices:
        y = frame.targets(i)
        if y is None:
            continue
        seq = []
        for j in range(i - SEQ_LEN + 1, i + 1):
            if j not in feats:
                feats[j] = frame.features(j) if j >= 0 else None
            if feats[j] is None:
                break
            seq.append(feats[j])
        if len(seq) == SEQ_LEN:
            X.append(seq), Y.append(y), kept.append(i)
    return X, Y, kept


FINETUNE_TRAIN_DAYS = 365
FINETUNE_VAL_DAYS = 28
# 첫 학습일 전 문맥 116일 + 학습 타깃 365일 + 겹침 방지 28일 + 검증 타깃 28일 + 마지막 검증의 4주 정답 28일 = 565
FINETUNE_MIN_ROWS = (
    GAP_WINDOW + SEQ_LEN - 2 + FINETUNE_TRAIN_DAYS + 7 * HORIZONS + FINETUNE_VAL_DAYS + 7 * HORIZONS
)


def split_finetune(
    frame: DailyFrame, train_days: int = FINETUNE_TRAIN_DAYS, val_days: int = FINETUNE_VAL_DAYS
) -> tuple[list[int], list[int]]:
    """fine-tuning (학습 날짜, 검증 날짜). 검증 = 정답이 확보된 가장 최근 val_days일.
    학습 = 그보다 28일 이상 앞선 train_days일 → 학습 타깃(다음 4주 평균)이 검증 날짜와 겹치지 않는다."""
    answered = [i for i in range(len(frame.dates)) if frame.targets(i) is not None]
    first_ok = GAP_WINDOW + SEQ_LEN - 2  # 입력 28일의 첫날도 90일 평균 피처가 있어야 함
    usable = [i for i in answered if i >= first_ok]
    if len(usable) < train_days + 7 * HORIZONS + val_days:
        raise ValueError(
            f"insufficient_data: fine-tuning에 최소 {FINETUNE_MIN_ROWS}행이 필요합니다 "
            f"(현재 {len(frame.dates)}행, 학습·검증 가능한 날 {len(usable)}일)"
        )
    val_idx = usable[-val_days:]
    train_idx = [i for i in usable if i <= val_idx[0] - 7 * HORIZONS - 1][-train_days:]
    return train_idx, val_idx
