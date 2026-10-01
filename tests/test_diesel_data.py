import csv
from datetime import date, timedelta

import pytest

from data.build_diesel_dataset import build_rows, write_rows
from data.diesel import DIESEL_FEATURE_COLUMNS, validate_diesel_rows
from data.external import asof_values, load_opinet_singapore_diesel, load_tax_policy
from data.opinet import load_opinet_diesel, load_opinet_diesel_many


def write_opinet(tmp_path, rows, header="구분,자동차용경유\r\n", encoding="cp949"):
    path = tmp_path / "diesel.csv"
    path.write_bytes((header + "".join(rows)).encode(encoding))
    return str(path)


ROWS = [
    "2026년09월28일,1842.11\r\n",
    "2026년09월29일,1843.25\r\n",
    "2026년09월30일,1843.86\r\n",
]


@pytest.mark.parametrize("encoding", ["cp949", "utf-8-sig"])
def test_converts_diesel_without_using_gasoline(tmp_path, encoding):
    path = write_opinet(tmp_path, ROWS, encoding=encoding)
    assert load_opinet_diesel(path) == [
        {"date": "2026-09-28", "diesel_price": 1842.11},
        {"date": "2026-09-29", "diesel_price": 1843.25},
        {"date": "2026-09-30", "diesel_price": 1843.86},
    ]
    assert load_opinet_diesel(path) == load_opinet_diesel(path)
    with pytest.raises(ValueError, match="자동차용경유"):
        load_opinet_diesel(write_opinet(tmp_path, ROWS, header="구분,보통휘발유\r\n"))


def test_diesel_duplicate_gap_and_invalid_values(tmp_path):
    assert len(load_opinet_diesel(write_opinet(tmp_path, [*ROWS, ROWS[-1]]))) == 3
    with pytest.raises(ValueError, match="중복"):
        load_opinet_diesel(write_opinet(tmp_path, [*ROWS, "2026년09월30일,1900\r\n"]))
    with pytest.raises(ValueError, match="누락"):
        load_opinet_diesel(write_opinet(tmp_path, [ROWS[0], ROWS[-1]]))
    for bad in ("0", "-1", "nan", "inf", ""):
        with pytest.raises(ValueError):
            load_opinet_diesel(write_opinet(tmp_path, [*ROWS[:2], f"2026년09월30일,{bad}\r\n"]))


def test_diesel_multiple_files_are_contiguous(tmp_path):
    first = write_opinet(tmp_path, ROWS[:2])
    second_path = tmp_path / "second.csv"
    second_path.write_bytes(("구분,자동차용경유\r\n" + ROWS[-1]).encode("cp949"))
    assert len(load_opinet_diesel_many([first, str(second_path)])) == 3
    second_path.write_bytes("구분,자동차용경유\r\n2026년10월01일,1844\r\n".encode("cp949"))
    with pytest.raises(ValueError, match="누락"):
        load_opinet_diesel_many([first, str(second_path)])


def test_singapore_price_skips_holidays_and_rejects_wrong_unit(tmp_path):
    first = tmp_path / "singapore_first.csv"
    second = tmp_path / "singapore_second.csv"
    first.write_bytes("기간,경유(0.001%)\r\n26년09월27일,95\r\n26년09월28일,\r\n".encode("cp949"))
    second.write_bytes("기간,경유(0.001%)\r\n26년09월29일,96\r\n".encode("cp949"))
    singapore = load_opinet_singapore_diesel([str(first), str(second)])
    assert asof_values([date(2026, 9, 28), date(2026, 9, 30)], singapore, 1) == [95, 96]
    second.write_bytes("기간,경유(0.001%)\r\n26년09월29일,818.38\r\n".encode("cp949"))
    with pytest.raises(ValueError, match="USD/bbl"):
        load_opinet_singapore_diesel([str(first), str(second)])


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        ("2018-11-05", 0.0),
        ("2018-11-06", 14.9),
        ("2019-05-06", 14.9),
        ("2019-05-07", 6.9),
        ("2019-08-31", 6.9),
        ("2019-09-01", 0.0),
        ("2021-11-11", 0.0),
        ("2021-11-12", 20.0),
        ("2022-04-30", 20.0),
        ("2022-05-01", 29.9),
        ("2022-06-30", 29.9),
        ("2022-07-01", 36.5),
        ("2023-09-29", 36.5),
        ("2023-09-30", 36.5),
        ("2024-06-30", 36.5),
        ("2024-07-01", 29.9),
        ("2024-10-31", 29.9),
        ("2024-11-01", 22.9),
        ("2025-04-30", 22.9),
        ("2025-05-01", 14.9),
        ("2025-10-31", 14.9),
        ("2025-11-01", 10.0),
        ("2026-03-31", 10.0),
        ("2026-04-01", 25.1),
        ("2026-09-30", 25.1),
    ],
)
def test_diesel_tax_boundaries(day, expected):
    from data.external import tax_cut_rates

    periods = load_tax_policy("data/reference/diesel_fuel_tax_cut.csv")
    assert tax_cut_rates([date.fromisoformat(day)], periods) == [expected]


def test_diesel_build_contract_and_write(tmp_path):
    diesel = [
        {"date": (date(2026, 9, 28) + timedelta(days=i)).isoformat(), "diesel_price": 1800 + i}
        for i in range(3)
    ]
    singapore = [(date(2026, 9, 27), 90.0), (date(2026, 9, 29), 95.0)]
    fx = [(date(2026, 9, 28), 1300.0), (date(2026, 9, 30), 1310.0)]
    rows, trimmed = build_rows(
        diesel, singapore, fx, load_tax_policy("data/reference/diesel_fuel_tax_cut.csv")
    )
    assert trimmed == 0
    assert list(rows[0]) == ["date", *DIESEL_FEATURE_COLUMNS]
    assert [r["singapore_diesel_price"] for r in rows] == [90.0, 90.0, 95.0]
    assert [r["usd_krw"] for r in rows] == [1300.0, 1300.0, 1310.0]
    assert [r["tax_or_supply_feature"] for r in rows] == [25.1] * 3
    assert all("gasoline_price" not in row for row in rows)

    path = tmp_path / "processed.csv"
    write_rows(rows, str(path))
    with path.open(encoding="utf-8", newline="") as f:
        saved = list(csv.DictReader(f))
    assert list(saved[0]) == ["date", *DIESEL_FEATURE_COLUMNS]
    assert len(saved) == 3


def test_diesel_build_uses_real_length_fx_holiday_without_future_value():
    days = [date(2017, 10, 7) + timedelta(days=i) for i in range(4)]
    diesel = [{"date": day.isoformat(), "diesel_price": 1500} for day in days]
    singapore = [(date(2017, 10, 6) + timedelta(days=i), 100 + i) for i in range(4)]
    fx = [(date(2017, 9, 29), 1300), (date(2017, 10, 10), 1310)]
    rows, trimmed = build_rows(
        diesel, singapore, fx, load_tax_policy("data/reference/diesel_fuel_tax_cut.csv")
    )
    assert trimmed == 0
    assert [row["usd_krw"] for row in rows] == [1300, 1300, 1300, 1310]


def test_diesel_validation_rejects_mixed_or_broken_rows():
    good = {
        "date": "2026-09-30",
        "diesel_price": 1843.86,
        "singapore_diesel_price": 95,
        "usd_krw": 1300,
        "tax_or_supply_feature": 25.1,
    }
    assert validate_diesel_rows([good])[0]["diesel_price"] == 1843.86
    for broken in (
        {**good, "diesel_price": ""},
        {**good, "diesel_price": "nan"},
        {**good, "usd_krw": -1},
        {**good, "date": "2026-09-31"},
    ):
        with pytest.raises(ValueError):
            validate_diesel_rows([broken])
    with pytest.raises(ValueError, match="diesel_price"):
        validate_diesel_rows([{k: v for k, v in good.items() if k != "diesel_price"}])
