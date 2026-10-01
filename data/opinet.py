"""
오피넷(www.opinet.co.kr) 원본 CSV를 공통 형식으로 변환하는 모듈.

오피넷 CSV저장 파일은 CP949 인코딩이고, 날짜가 "2026년09월26일"(국내 평균판매가격)이나
"26년09월24일"(국제유가)처럼 한글 형식으로 들어 있습니다. 여기서 ISO 날짜와 숫자로
정규화합니다. 보통휘발유와 자동차용경유는 서로 다른 목표 컬럼으로 반환합니다.

원본 파일은 data/raw/에 그대로 보존하고(커밋하지 않음), 변환은 항상 원본에서 다시
수행해 같은 원본이면 같은 결과가 나오도록 합니다.
"""

import csv
import io
from datetime import date, datetime, timedelta
from math import isfinite

GASOLINE_COLUMN = "보통휘발유"
DIESEL_COLUMN = "자동차용경유"
DATE_FORMATS = ("%Y년%m월%d일", "%y년%m월%d일", "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y%m%d")


def read_csv_text(path: str) -> list[dict]:
    """오피넷 원본(CP949)과 재저장본(UTF-8)을 모두 읽는다."""
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"{path}: UTF-8/CP949로 읽을 수 없는 파일입니다")
    return list(csv.DictReader(io.StringIO(text)))


def parse_date(value: str) -> date:
    text = value.strip().replace(" ", "")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"날짜 형식을 해석할 수 없습니다: {value!r}")


def parse_number(value: str) -> float:
    number = float(str(value).strip().replace(",", ""))
    if not isfinite(number):
        raise ValueError(f"유한한 숫자가 아닙니다: {value!r}")
    return number


def parse_series(rows: list[dict], date_column: str, value_column: str) -> list[tuple[date, float]]:
    """(날짜, 값) 목록을 날짜순으로 반환. 같은 날짜가 같은 값으로 중복되면 하나만 남기고,
    값이 다르면 어느 쪽이 맞는지 알 수 없으므로 오류로 처리한다."""
    if not rows:
        raise ValueError("데이터 행이 없습니다")
    for column in (date_column, value_column):
        if column not in rows[0]:
            raise ValueError(f"{column!r} 컬럼이 없습니다 (컬럼: {list(rows[0])})")

    by_day: dict[date, float] = {}
    for row in rows:
        day = parse_date(row[date_column])
        value = parse_number(row[value_column])
        if day in by_day and by_day[day] != value:
            raise ValueError(f"{day} 날짜가 서로 다른 값으로 중복됩니다")
        by_day[day] = value
    return sorted(by_day.items())


def _load_opinet_product(path: str, column: str, price_key: str) -> list[dict]:
    """제품별 일간 평균판매가격을 날짜와 원/L 가격으로 변환한다."""
    series = parse_series(read_csv_text(path), "구분", column)
    for (day, _), (next_day, _) in zip(series, series[1:]):
        if next_day != day + timedelta(days=1):
            raise ValueError(
                f"{day} 다음 날짜가 {next_day}입니다 (누락된 날짜는 보간하지 않습니다)"
            )
    for day, price in series:
        if price <= 0:
            raise ValueError(f"{day} {column} 가격이 양수가 아닙니다: {price}")
    return [{"date": day.isoformat(), price_key: price} for day, price in series]


def load_opinet_gasoline(path: str, column: str = GASOLINE_COLUMN) -> list[dict]:
    """오피넷 일간 보통휘발유 CSV -> date, gasoline_price (기존 휘발유 경로 전용, 경유 계약과 분리)."""
    return _load_opinet_product(path, column, "gasoline_price")


def load_opinet_diesel(path: str, column: str = DIESEL_COLUMN) -> list[dict]:
    """오피넷 일간 자동차용경유 CSV -> date, diesel_price."""
    return _load_opinet_product(path, column, "diesel_price")


def load_opinet_diesel_many(paths: list[str]) -> list[dict]:
    """기간별로 나눠 받은 경유 CSV를 이어 붙이고 경계의 중복·누락을 검사한다."""
    if not paths:
        raise ValueError("경유 원본 파일이 없습니다")
    by_day: dict[str, float] = {}
    for path in paths:
        for row in load_opinet_diesel(path):
            day, price = row["date"], row["diesel_price"]
            if day in by_day and by_day[day] != price:
                raise ValueError(f"{day} 경유 가격이 원본 파일 사이에서 다릅니다")
            by_day[day] = price
    days = sorted(by_day)
    for previous, current in zip(days, days[1:]):
        if date.fromisoformat(current) != date.fromisoformat(previous) + timedelta(days=1):
            raise ValueError(f"{previous} 다음 날짜가 {current}입니다 (경유 가격 누락)")
    return [{"date": day, "diesel_price": by_day[day]} for day in days]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="오피넷 휘발유 평균판매가격 CSV 변환 확인")
    parser.add_argument("path", help="오피넷 CSV저장 원본 파일 (data/raw/)")
    args = parser.parse_args()

    raw_rows = read_csv_text(args.path)
    rows = load_opinet_gasoline(args.path)
    prices = [row["gasoline_price"] for row in rows]
    print(f"원본 {len(raw_rows)}행 -> 변환 {len(rows)}행")
    print(
        f"기간 {rows[0]['date']} ~ {rows[-1]['date']}, 가격 {min(prices):.2f} ~ {max(prices):.2f}원/L"
    )


if __name__ == "__main__":
    main()
