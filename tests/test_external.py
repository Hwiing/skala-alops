from datetime import date, timedelta

import pytest

from data.build_dataset import build_rows
from data.ecos import parse_ecos_response
from data.external import (
    asof_values,
    load_opinet_crude,
    load_tax_policy,
    load_usd_krw,
    tax_cut_rates,
)
from data.features import FEATURE_COLUMNS

D = date.fromisoformat


def write(tmp_path, name, text, encoding="cp949"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return str(path)


# 2023-01-06(금) ~ 2023-01-11(수). 주말(7, 8일)에는 국제유가·환율 값이 없다.
CRUDE_CSV = (
    "기간,Dubai,Brent,WTI\r\n"
    "23년01월05일,77.00,78.69,73.67\r\n"
    "23년01월06일,76.00,78.57,73.77\r\n"
    "23년01월09일,78.00,79.65,74.63\r\n"
    "23년01월10일,79.00,80.10,75.12\r\n"
    "23년01월11일,80.00,82.67,77.41\r\n"
)
FX_CSV = (
    "date,usd_krw\n"
    "2023-01-05,1274.7\n2023-01-06,1268.2\n2023-01-09,1244.4\n2023-01-10,1238.5\n2023-01-11,1245.2\n"
)


def gasoline(start="2023-01-06", days=6):
    return [
        {"date": (D(start) + timedelta(days=i)).isoformat(), "gasoline_price": 1540.0 + i}
        for i in range(days)
    ]


def test_crude_uses_previous_trading_day_only(tmp_path):
    crude = load_opinet_crude(write(tmp_path, "crude.csv", CRUDE_CSV))
    days = [D("2023-01-06"), D("2023-01-07"), D("2023-01-08"), D("2023-01-09"), D("2023-01-10")]
    # 금요일은 목요일 값, 주말·월요일은 금요일 값, 화요일은 월요일 값
    assert asof_values(days, crude, lag_days=1) == [77.0, 76.0, 76.0, 76.0, 78.0]


def test_singapore_holiday_blank_is_skipped_not_zero(tmp_path):
    # 실제 원본: 싱가포르 공휴일(예: 23년11월13일 디파발리)에는 Dubai 값이 비어 있다
    text = "기간,Dubai\r\n23년11월10일,79.00\r\n23년11월13일,\r\n23년11월14일,81.00\r\n"
    crude = load_opinet_crude(write(tmp_path, "crude.csv", text))
    assert crude == [(D("2023-11-10"), 79.0), (D("2023-11-14"), 81.0)]
    assert asof_values([D("2023-11-14")], crude, lag_days=1) == [79.0]


def test_rejects_crude_saved_in_won_per_liter(tmp_path):
    text = "기간,Dubai\r\n23년09월28일,818.38\r\n23년09월29일,812.88\r\n"
    with pytest.raises(ValueError, match=r"`\$`"):
        load_opinet_crude(write(tmp_path, "crude.csv", text))


def test_fx_uses_same_day_and_carries_weekend_forward(tmp_path):
    fx = load_usd_krw(write(tmp_path, "fx.csv", FX_CSV, "utf-8"))
    days = [D("2023-01-06"), D("2023-01-07"), D("2023-01-08"), D("2023-01-09")]
    assert asof_values(days, fx, lag_days=0) == [1268.2, 1268.2, 1268.2, 1244.4]


def test_future_observations_never_change_past_features(tmp_path):
    crude = load_opinet_crude(write(tmp_path, "crude.csv", CRUDE_CSV))
    days = [D("2023-01-06") + timedelta(days=i) for i in range(4)]
    before = asof_values(days, crude, lag_days=1)
    shocked = [(day, price * 3 if day >= D("2023-01-09") else price) for day, price in crude]
    assert asof_values(days, shocked, lag_days=1) == before


def test_rejects_stale_or_missing_history():
    series = [(D("2023-01-02"), 80.0)]
    with pytest.raises(ValueError, match="오래됐습니다"):
        asof_values([D("2023-01-20")], series, lag_days=0)
    with pytest.raises(ValueError, match="관측값이 없습니다"):
        asof_values([D("2023-01-01")], series, lag_days=0)


@pytest.mark.parametrize(
    ("day", "rate"),
    [
        ("2021-11-11", 0.0),
        ("2021-11-12", 20.0),
        ("2022-07-01", 37.1),
        ("2023-01-01", 25.0),
        ("2024-06-30", 25.0),
        ("2024-07-01", 20.0),
        ("2024-11-01", 14.9),
        ("2025-05-01", 10.0),
        ("2025-11-01", 7.0),
        ("2026-03-31", 7.0),
        ("2026-04-01", 14.9),
        ("2026-11-30", 14.9),
    ],
)
def test_tax_cut_rate_boundaries(day, rate):
    assert tax_cut_rates([D(day)], load_tax_policy()) == [rate]


def test_tax_cut_rate_matches_statutory_won_per_liter():
    # 인하율 = 1 - 탄력세율 / 기본 529원/L (시행령 개정이유의 원/L 값에서 계산)
    import csv

    with open("data/reference/gasoline_fuel_tax_cut.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expected = round((1 - float(row["flexible_tax_krw_per_l"]) / 529) * 100, 1)
            assert float(row["cut_rate_pct"]) == expected, row["start_date"]


def test_tax_rate_outside_verified_table_is_an_error():
    with pytest.raises(ValueError, match="유류세"):
        tax_cut_rates([D("2026-12-01")], load_tax_policy())


def test_tax_table_must_be_contiguous(tmp_path):
    path = tmp_path / "tax.csv"
    path.write_text(
        "start_date,end_date,cut_rate_pct,source\n"
        "2023-01-01,2023-01-31,25,a\n2023-02-02,2023-02-28,25,b\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="이어지지"):
        load_tax_policy(str(path))


def test_build_rows_matches_contract(tmp_path):
    crude = load_opinet_crude(write(tmp_path, "crude.csv", CRUDE_CSV))
    fx = load_usd_krw(write(tmp_path, "fx.csv", FX_CSV, "utf-8"))
    rows, trimmed = build_rows(gasoline("2023-01-05"), crude, fx, load_tax_policy())
    # 2023-01-05는 그 전날 공개된 국제유가가 없으므로 추정하지 않고 제외
    assert trimmed == 1
    assert rows[0]["date"] == "2023-01-06"
    assert list(rows[0]) == ["date", *FEATURE_COLUMNS]
    assert rows[0] == {
        "date": "2023-01-06",
        "gasoline_price": 1541.0,
        "crude_oil_price": 77.0,
        "usd_krw": 1268.2,
        "tax_or_supply_feature": 25.0,
    }


def test_parse_ecos_response():
    payload = {
        "StatisticSearch": {
            "list_total_count": 2,
            "row": [
                {"TIME": "20230102", "DATA_VALUE": "1267.3", "ITEM_CODE1": "0000001"},
                {"TIME": "20230103", "DATA_VALUE": "1268.9", "ITEM_CODE1": "0000001"},
            ],
        }
    }
    assert parse_ecos_response(payload) == [("2023-01-02", 1267.3), ("2023-01-03", 1268.9)]
    with pytest.raises(ValueError, match="ECOS"):
        parse_ecos_response(
            {"RESULT": {"CODE": "INFO-200", "MESSAGE": "해당하는 데이터가 없습니다."}}
        )
