import csv
from datetime import date, timedelta

import pytest

from data.build_extended_diesel_dataset import (
    calculate_spread,
    provenance_rows,
    splice_singapore,
)
from data.external import load_ecos_wide_usd_krw, load_tax_policy, tax_cut_rates


def test_ecos_wide_extracts_only_usd_krw_and_checks_unit(tmp_path):
    path = tmp_path / "ecos.csv"
    path.write_text(
        "통계표,계정항목,단위,변환,2008/04/14,2008/04/15,2008/04/16\n"
        "3.1.1.1,원/일본엔(100엔),원,원자료,960,970,980\n"
        '3.1.1.1,원/미국달러(매매기준율),원,원자료,"1,000.20",,998.50\n',
        encoding="utf-8-sig",
    )
    assert load_ecos_wide_usd_krw(str(path)) == [
        (date(2008, 4, 14), 1000.2),
        (date(2008, 4, 16), 998.5),
    ]
    path.write_text("계정항목,단위,2008/04/14\n원/미국달러(매매기준율),달러,1000\n")
    with pytest.raises(ValueError, match="단위 원"):
        load_ecos_wide_usd_krw(str(path))


def test_spread_uses_first_60_paired_market_days_and_marks_only_estimated_dates(tmp_path):
    path = tmp_path / "overlap.csv"
    first = date(2012, 12, 3)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["기간", "경유(0.001%)", "경유(0.05%)"])
        for i in range(61):
            day = first + timedelta(days=i)
            writer.writerow([day.isoformat(), 101.5 if i < 60 else 200, 100])
    assert calculate_spread(str(path)) == 1.5

    low = [(date(2012, 11, 30), 100.0)]
    high = [(first, 102.0)]
    series, transition = splice_singapore(low, high, 1.5)
    assert series == [(date(2012, 11, 30), 101.5), (first, 102.0)]
    rows = [{"date": "2012-12-03"}, {"date": "2012-12-04"}]
    flags = provenance_rows(rows, series, transition)
    assert [row["singapore_diesel_spliced"] for row in flags] == [True, False]
    assert [row["singapore_price_observation_date"] for row in flags] == [
        "2012-11-30",
        "2012-12-03",
    ]


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        ("2008-04-15", 10.7),
        ("2008-10-06", 10.7),
        ("2008-10-07", 12.5),
        ("2008-12-31", 12.5),
        ("2009-01-01", 2.9),
        ("2009-05-20", 2.9),
        ("2009-05-21", 0.0),
        ("2012-11-30", 0.0),
        ("2012-12-01", 0.0),
    ],
)
def test_historical_diesel_tax_boundaries(day, expected):
    periods = load_tax_policy("data/reference/diesel_fuel_tax_cut.csv")
    assert tax_cut_rates([date.fromisoformat(day)], periods) == [expected]
