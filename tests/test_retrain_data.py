"""재학습 데이터 선택: 업로드 시각보다 데이터 마지막 날짜·검증·최소 행 수가 우선이다."""

import csv

import pytest

from data import storage
from serving_app.monitoring import retrain_trigger as rt
from tests.test_scaffold import make_rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def data_files(tmp_path, monkeypatch):
    base = tmp_path / "base.csv"
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(rt, "DATA_CSV", str(base))
    monkeypatch.setattr(storage, "UPLOAD_DIR", str(uploads))
    return base, uploads


def test_newer_complete_upload_is_used_and_fingerprint_logged(data_files, caplog):
    base, uploads = data_files
    write_csv(base, make_rows(700, "2023-01-01"))
    rows = make_rows(700, "2024-01-01")
    write_csv(uploads / "new.csv", rows)
    with caplog.at_level("INFO", logger="aiops"):
        assert rt._load_recent_rows() == rows[-627:]
    assert "new.csv" in caplog.text and "sha256=" in caplog.text


@pytest.mark.parametrize("kind", ["short", "old", "invalid", "pending"])
def test_ineligible_upload_falls_back_to_configured_csv(data_files, kind):
    base, uploads = data_files
    expected = make_rows(700, "2024-01-01")
    write_csv(base, expected)
    if kind == "short":
        write_csv(uploads / "new.csv", make_rows(175, "2026-01-01"))
    elif kind == "old":
        write_csv(uploads / "old.csv", make_rows(700, "2023-01-01"))
    elif kind == "invalid":
        (uploads / "bad.csv").write_text("not a valid csv")
    else:
        write_csv(uploads / "new.csv.pending", make_rows(700, "2026-01-01"))
    assert rt._load_recent_rows() == expected[-627:]


def test_all_uploads_are_considered_even_when_last_uploaded_is_short(data_files):
    _, uploads = data_files
    expected = make_rows(700, "2024-01-01")
    write_csv(uploads / "a.csv", expected)
    write_csv(uploads / "z.csv", make_rows(175, "2026-01-01"))
    assert rt._load_recent_rows() == expected[-627:]


def test_equal_end_dates_prefer_configured_source(data_files):
    base, uploads = data_files
    rows = make_rows(700)
    write_csv(base, rows)
    different = [dict(r, diesel_price=2000) for r in rows]
    write_csv(uploads / "same.csv", different)
    assert rt._load_recent_rows() == rows[-627:]


def test_no_complete_source_reports_insufficient_data(data_files):
    _, uploads = data_files
    write_csv(uploads / "short.csv", make_rows(175))
    with pytest.raises(ValueError, match="insufficient_data"):
        rt._load_recent_rows()
