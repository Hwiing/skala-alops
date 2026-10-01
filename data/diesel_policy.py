"""
경유 정책표(유류세·석유 최고가격)를 날짜별 값으로 바꾸는 순수 파이썬 유틸리티.

모델은 정책 효과를 직접 배우지 않고 "세금 효과를 뺀 가격"의 변화를 예측한다. 사전 고시되는 정책은
예측 후 규칙으로 더한다(정책 규칙, contracts.md v2). 정책이 바뀌면 data/reference 표만 고치면 된다.

- 유류세(원/L, VAT 제외) = 교통·에너지·환경세 탄력세율 × (1 + 교육세 15% + 주행세율)
  주행세율은 지방세법 시행령 개정이력(대통령령 제20746·21067·21498호)에 따라 기간별로 다르다.
- 소매가 반영 시차: 세금·상한 변경은 주유소 재고 때문에 당일 33%만 반영되고 14일에 걸쳐 100% 반영된다
  (2009~2026 경유세 변경 12건 이벤트 스터디, docs/evidence/06 §12). 시차 일수는 2012~2025 성적으로만 골랐다.
- 최고가격(2026-03-13~)은 정유사 공급가 세후 상한이다. 상한 차수는 발표일부터 알려진 것으로 보고,
  아직 발표되지 않은 기간은 현재 상한이 이어진다고 가정한다.
"""

import csv
from bisect import bisect_right
from datetime import date

TAX_PATH = "data/reference/diesel_fuel_tax_cut.csv"
# 세율 변경 발표일 (시행일 → 처음 공식 발표일). 세금표는 데이터 담당과 공유하므로 발표일은 따로 둔다.
TAX_ANNOUNCED_PATH = "data/reference/diesel_tax_announced.csv"
CAP_PATH = "data/reference/price_cap.csv"
EDUCATION_TAX = 0.15
# (시행일, 주행세율): 2008-03-10 32.5%→27%, 2008-10-07 →30%, 2009-05-21 →26%(2011년부터 자동차세 주행분)
DRIVING_TAX = [
    (date(1900, 1, 1), 0.325),
    (date(2008, 3, 10), 0.27),
    (date(2008, 10, 7), 0.30),
    (date(2009, 5, 21), 0.26),
]
PASS_DAY0 = 0.33  # 변경 당일 소매가 반영률
PASS_DAYS = 14  # 100% 반영까지 걸리는 일수


def _day(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def driving_rate(day: date) -> float:
    starts = [start for start, _ in DRIVING_TAX]
    return DRIVING_TAX[bisect_right(starts, day) - 1][1]


def pass_ratio(days_since: int) -> float:
    """변경 후 days_since일째 소매가에 반영된 비율 (변경 전 0, 당일 0.33, 14일째 1)."""
    if days_since < 0:
        return 0.0
    return min(1.0, PASS_DAY0 + (1 - PASS_DAY0) * days_since / PASS_DAYS)


class TaxSchedule:
    """유류세 구간표. tax(day) = 그날 시행 세액, retail_tax(day) = 시차를 반영해 소매가에 들어간 세액."""

    def __init__(self, path: str = TAX_PATH, announced_path: str | None = TAX_ANNOUNCED_PATH):
        with open(path, encoding="utf-8-sig") as f:
            rows = sorted(csv.DictReader(f), key=lambda r: r["start_date"])
        self.starts = [_day(r["start_date"]) for r in rows]
        announced = {}
        if announced_path:
            with open(announced_path, encoding="utf-8-sig") as f:
                announced = {
                    _day(r["start_date"]): _day(r["announced_date"]) for r in csv.DictReader(f)
                }
        # 발표일이 없으면 시행일에 알려진 것으로 본다
        self.announced = [announced.get(d, d) for d in self.starts]
        self.values = [
            float(r["flexible_tax_krw_per_l"])
            * (1 + EDUCATION_TAX + driving_rate(_day(r["start_date"])))
            for r in rows
        ]

    def tax(self, day) -> float:
        i = bisect_right(self.starts, _day(day)) - 1
        if i < 0:
            raise ValueError(f"유류세 표에 {day} 이전 구간이 없습니다")
        return self.values[i]

    def retail_tax(self, day, as_of=None) -> float:
        """day의 소매가에 들어간 세액. as_of를 주면 그날까지 발표된 변경만 반영한다(미발표 변경은 현재 세율 유지)."""
        day = _day(day)
        as_of = _day(as_of) if as_of is not None else None
        i = bisect_right(self.starts, day) - 1
        if i < 0:
            raise ValueError(f"유류세 표에 {day} 이전 구간이 없습니다")
        value = prev = self.values[0]
        for j in range(1, i + 1):
            if as_of is not None and self.announced[j] > as_of:
                continue
            value += (self.values[j] - prev) * pass_ratio((day - self.starts[j]).days)
            prev = self.values[j]
        return value


class CapSchedule:
    """경유 최고가격 구간표 (세후 원/L). 발표일 기준으로 '그 시점에 알려진' 상한을 돌려준다."""

    def __init__(self, path: str = CAP_PATH):
        with open(path, encoding="utf-8-sig") as f:
            self.rows = [
                (
                    _day(r["start_date"]),
                    _day(r["end_date"]),
                    float(r["diesel_cap_won_l"]),
                    _day(r["announced_date"]),
                )
                for r in csv.DictReader(f)
            ]

    def cap(self, day) -> float | None:
        day = _day(day)
        return next((c for s, e, c, _ in self.rows if s <= day <= e), None)

    def _known(self, as_of: date) -> list[tuple]:
        """as_of까지 발표된 상한 구간 (시행일 순)."""
        return sorted((r for r in self.rows if r[3] <= as_of), key=lambda r: r[0])

    def known_cap(self, day, as_of) -> float | None:
        """as_of 시점에 알려진 day의 상한. 발표된 마지막 구간 뒤로는 그 상한이 이어진다고 본다."""
        day, as_of = _day(day), _day(as_of)
        started = [c for s, _, c, _ in self._known(as_of) if s <= day]
        return started[-1] if started else None

    def _retail_level(self, day: date, known: list[tuple]) -> float:
        """알려진 상한 경로가 day의 소매가에 반영된 누적액 (변경마다 재고 시차 적용)."""
        level, prev = 0.0, None
        for s, _, c, _ in known:
            if s > day:
                break
            level += (c if prev is None else c - prev) * (
                1.0 if prev is None else pass_ratio((day - s).days)
            )
            prev = c
        return level

    def retail_change(self, day, as_of) -> float:
        """as_of 대비 day에 소매가로 더 반영될 상한 변화분.

        발표된 변경만 쓰고, 이미 시행돼 반영이 진행 중인 변경의 남은 몫도 포함한다
        (= day의 누적 반영액 − as_of의 누적 반영액).
        """
        day, as_of = _day(day), _day(as_of)
        known = self._known(as_of)
        if not known or known[0][0] > as_of:  # as_of에 상한이 없으면 묶일 수 없다
            return 0.0
        return self._retail_level(day, known) - self._retail_level(as_of, known)
