"""
외부 피처(국제유가·환율·유류세 인하율)를 휘발유 날짜에 맞춰 붙이는 모듈.

핵심 원칙은 "D일 입력에는 D일 예측 시점에 이미 공개된 값만 쓴다"입니다.
국제유가·환율은 거래일에만 값이 있으므로, 각 D일에 대해 공개 시점(lag)을 지난 가장
최근 관측값을 가져옵니다(as-of). 뒤 날짜 값으로 채우는 보간은 하지 않습니다.

    crude_oil_price : 오피넷 두바이유 현물(USD/bbl). 싱가포르 장 마감 기준이라 같은 날
                      국내 가격 집계보다 늦게 공개될 수 있어 1일 지연(D-1까지)만 사용.
    usd_krw         : 한국은행 ECOS 원/미국달러 매매기준율. D일 오전 고시되므로 D일 값 사용.
    tax_or_supply_feature : 휘발유 유류세 인하율(%). 시행일이 사전 고시되는 정책값.
                      공급 차질은 일별 관측 지표가 없어 피처에서 제외하고 드리프트 시나리오로만 다룬다.
"""

import csv
from bisect import bisect_right
from datetime import date, timedelta

from data.opinet import parse_date, parse_number, parse_series, read_csv_text

CRUDE_COLUMN = "Dubai"
CRUDE_LAG_DAYS = 1
FX_LAG_DAYS = 0
# 설·추석 연휴에도 국제유가·환율 공백은 일주일을 넘지 않는다. 넘으면 원본 누락으로 본다.
MAX_STALE_DAYS = 7
TAX_POLICY_PATH = "data/reference/gasoline_fuel_tax_cut.csv"


def load_opinet_crude(path: str, column: str = CRUDE_COLUMN) -> list[tuple[date, float]]:
    """오피넷 '유가관련정보 > 국제유가 > 원유' CSV저장($ 단위) -> [(날짜, USD/bbl)]."""
    series = parse_series(read_csv_text(path), "기간", column)
    for day, price in series:
        if price <= 0:
            raise ValueError(f"{day} 국제유가가 양수가 아닙니다: {price}")
    return series


def load_usd_krw(path: str) -> list[tuple[date, float]]:
    """data/ecos.py가 저장한 date,usd_krw CSV -> [(날짜, KRW/USD)]."""
    series = parse_series(read_csv_text(path), "date", "usd_krw")
    for day, rate in series:
        if rate <= 0:
            raise ValueError(f"{day} 환율이 양수가 아닙니다: {rate}")
    return series


def load_tax_policy(path: str = TAX_POLICY_PATH) -> list[tuple[date, date, float]]:
    """유류세 인하율 구간표. 구간은 빈틈·겹침 없이 이어져야 한다."""
    with open(path, encoding="utf-8") as f:
        periods = [
            (
                parse_date(r["start_date"]),
                parse_date(r["end_date"]),
                parse_number(r["cut_rate_pct"]),
            )
            for r in csv.DictReader(f)
        ]
    periods.sort()
    for (_, end, _), (next_start, _, _) in zip(periods, periods[1:]):
        if next_start != end + timedelta(days=1):
            raise ValueError(f"유류세 구간이 {end} 이후 {next_start}로 이어지지 않습니다")
    return periods


def tax_cut_rates(days: list[date], periods: list[tuple[date, date, float]]) -> list[float]:
    """구간표 밖의 날짜는 확인되지 않은 정책이므로 추정하지 않고 오류로 알린다."""
    rates = []
    for day in days:
        rate = next((r for start, end, r in periods if start <= day <= end), None)
        if rate is None:
            raise ValueError(f"{day} 유류세 인하율이 {TAX_POLICY_PATH}에 없습니다")
        rates.append(rate)
    return rates


def asof_values(
    days: list[date],
    series: list[tuple[date, float]],
    lag_days: int,
    max_stale_days: int = MAX_STALE_DAYS,
) -> list[float]:
    """각 날짜 D에 대해 D - lag_days 이전의 가장 최근 관측값. 미래 값은 절대 쓰지 않는다."""
    obs_days = [day for day, _ in series]
    values = []
    for day in days:
        cutoff = day - timedelta(days=lag_days)
        i = bisect_right(obs_days, cutoff) - 1
        if i < 0:
            raise ValueError(f"{day}: {cutoff} 이전 관측값이 없습니다")
        if (cutoff - obs_days[i]).days > max_stale_days:
            raise ValueError(
                f"{day}: 마지막 관측값 {obs_days[i]}이 {max_stale_days}일보다 오래됐습니다"
            )
        values.append(series[i][1])
    return values


def first_available_day(series: list[tuple[date, float]], lag_days: int) -> date:
    return series[0][0] + timedelta(days=lag_days)
