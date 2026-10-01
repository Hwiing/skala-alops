import pytest

from data.opinet import load_opinet_gasoline, parse_date

# 오피넷 '평균판매가격 > 제품별 > 일간' CSV저장 파일과 같은 형식 (CP949, CRLF, 주말 포함)
HEADER = "구분,고급휘발유,보통휘발유,자동차용경유,실내등유\r\n"
ROWS = [
    "2026년09월26일,2332.32,1856.05,1841.44,1573.86\r\n",
    "2026년09월27일,2332.84,1856.05,1841.28,1573.84\r\n",
    "2026년09월28일,2335.12,1858.22,1843.81,1573.48\r\n",
    "2026년09월29일,2337.67,1858.18,1843.86,1573.54\r\n",
]


def write(tmp_path, rows, encoding="cp949"):
    path = tmp_path / "opinet.csv"
    path.write_bytes((HEADER + "".join(rows)).encode(encoding))
    return str(path)


@pytest.mark.parametrize("encoding", ["cp949", "utf-8-sig"])
def test_converts_opinet_csv(tmp_path, encoding):
    rows = load_opinet_gasoline(write(tmp_path, ROWS, encoding))
    assert rows[0] == {"date": "2026-09-26", "gasoline_price": 1856.05}
    assert [row["date"] for row in rows] == [
        "2026-09-26",
        "2026-09-27",
        "2026-09-28",
        "2026-09-29",
    ]


def test_same_input_gives_same_output(tmp_path):
    path = write(tmp_path, ROWS)
    assert load_opinet_gasoline(path) == load_opinet_gasoline(path)


def test_sorts_and_drops_identical_duplicates(tmp_path):
    rows = load_opinet_gasoline(write(tmp_path, [ROWS[2], ROWS[0], ROWS[1], ROWS[3], ROWS[3]]))
    assert len(rows) == 4
    assert rows[0]["date"] == "2026-09-26"


def test_rejects_conflicting_duplicate(tmp_path):
    conflict = "2026년09월29일,2337.67,1900.00,1843.86,1573.54\r\n"
    with pytest.raises(ValueError, match="중복"):
        load_opinet_gasoline(write(tmp_path, [*ROWS, conflict]))


def test_rejects_missing_date_without_interpolation(tmp_path):
    with pytest.raises(ValueError, match="누락"):
        load_opinet_gasoline(write(tmp_path, [ROWS[0], ROWS[2]]))


@pytest.mark.parametrize("price", ["0", "-1", "nan", "inf", ""])
def test_rejects_invalid_price(tmp_path, price):
    bad = f"2026년09월30일,2337.67,{price},1843.86,1573.54\r\n"
    with pytest.raises(ValueError):
        load_opinet_gasoline(write(tmp_path, [*ROWS, bad]))


def test_rejects_missing_gasoline_column(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_bytes("구분,자동차용경유\r\n2026년09월26일,1841.44\r\n".encode("cp949"))
    with pytest.raises(ValueError, match="보통휘발유"):
        load_opinet_gasoline(str(path))


def test_parses_two_digit_year_used_by_crude_page():
    assert parse_date("26년09월24일").isoformat() == "2026-09-24"
