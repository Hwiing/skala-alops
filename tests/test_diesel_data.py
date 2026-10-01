import csv
from datetime import date, timedelta

import pytest

from data.build_diesel_dataset import build_rows, write_rows
from data.diesel import DIESEL_FEATURE_COLUMNS, validate_diesel_rows
from data.external import load_tax_policy
from data.opinet import load_opinet_diesel


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


@pytest.mark.parametrize(
    ("day", "expected"),
    [
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
    crude = [(date(2026, 9, 27), 90.0), (date(2026, 9, 29), 95.0)]
    fx = [(date(2026, 9, 28), 1300.0), (date(2026, 9, 30), 1310.0)]
    rows, trimmed = build_rows(
        diesel, crude, fx, load_tax_policy("data/reference/diesel_fuel_tax_cut.csv")
    )
    assert trimmed == 0
    assert list(rows[0]) == ["date", *DIESEL_FEATURE_COLUMNS]
    assert [r["crude_oil_price"] for r in rows] == [90.0, 90.0, 95.0]
    assert [r["usd_krw"] for r in rows] == [1300.0, 1300.0, 1310.0]
    assert [r["tax_or_supply_feature"] for r in rows] == [25.1] * 3
    assert all("gasoline_price" not in row for row in rows)

    path = tmp_path / "processed.csv"
    write_rows(rows, str(path))
    with path.open(encoding="utf-8", newline="") as f:
        saved = list(csv.DictReader(f))
    assert list(saved[0]) == ["date", *DIESEL_FEATURE_COLUMNS]
    assert len(saved) == 3


def test_diesel_validation_rejects_mixed_or_broken_rows():
    good = {
        "date": "2026-09-30",
        "diesel_price": 1843.86,
        "crude_oil_price": 95,
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
