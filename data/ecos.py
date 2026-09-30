"""
한국은행 ECOS Open API에서 원/미국달러 매매기준율(일별)을 받아 data/raw/에 저장한다.

통계표 731Y001(3.1.1.1 주요국 통화의 대원화환율), 항목 0000001(원/미국달러 매매기준율).
인증키는 https://ecos.bok.or.kr/api/ 에서 무료로 발급받아 환경변수로 전달합니다.

    ECOS_API_KEY=... .venv/bin/python -m data.ecos --start 20201201 --end 20231231
"""

import csv
import json
import os
from urllib.request import urlopen

from data.opinet import parse_date, parse_number

ECOS_URL = "https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/100000/731Y001/D/{start}/{end}/0000001"
DEFAULT_OUT = "data/raw/ecos_usdkrw.csv"


def parse_ecos_response(payload: dict) -> list[tuple[str, float]]:
    if "StatisticSearch" not in payload:
        # 오류 응답 예: {"RESULT": {"CODE": "INFO-200", "MESSAGE": "해당하는 데이터가 없습니다."}}
        raise ValueError(f"ECOS 오류 응답: {payload.get('RESULT', payload)}")
    rows = payload["StatisticSearch"]["row"]
    return [(parse_date(r["TIME"]).isoformat(), parse_number(r["DATA_VALUE"])) for r in rows]


def fetch_usd_krw(api_key: str, start: str, end: str) -> list[tuple[str, float]]:
    with urlopen(ECOS_URL.format(key=api_key, start=start, end=end), timeout=30) as resp:
        return parse_ecos_response(json.load(resp))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ECOS 원/달러 매매기준율 다운로드")
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    api_key = os.environ.get("ECOS_API_KEY")
    if not api_key:
        raise SystemExit("ECOS_API_KEY 환경변수가 필요합니다 (https://ecos.bok.or.kr/api/)")
    rows = fetch_usd_krw(api_key, args.start, args.end)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "usd_krw"])
        writer.writerows(rows)
    print(f"{len(rows)}행 저장 ({rows[0][0]} ~ {rows[-1][0]}) -> {args.out}")


if __name__ == "__main__":
    main()
