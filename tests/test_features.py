import os
from datetime import date, timedelta

import pytest

from data import storage
from data.features import (
    FEATURE_COLUMNS,
    SEQ_LEN,
    GasolineScaler,
    build_sequences,
    scaler_fit_rows,
    split_recent_for_finetune,
    train_test_split,
    validate_rows,
)


def make_rows(n=60, start="2023-01-01"):
    """가격이 날짜마다 1원씩 오르는 fixture - 타깃 정렬을 값으로 바로 확인할 수 있다."""
    return [
        {
            "date": (date.fromisoformat(start) + timedelta(days=i)).isoformat(),
            "diesel_price": str(1500 + i),
            "singapore_diesel_price": str(80 + i / 10),
            "usd_krw": "1300",
            "tax_or_supply_feature": "25",
        }
        for i in range(n)
    ]


def test_feature_order_is_the_shared_contract():
    assert FEATURE_COLUMNS == (
        "diesel_price",
        "singapore_diesel_price",
        "usd_krw",
        "tax_or_supply_feature",
    )


def test_sequences_are_n_by_20_by_4_and_target_is_next_day():
    rows = validate_rows(make_rows())
    X, y = build_sequences(rows, GasolineScaler().fit(rows))
    assert (len(X), len(X[0]), len(X[0][0])) == (60 - SEQ_LEN, SEQ_LEN, 4)
    for i in (0, len(X) - 1):
        # i번째 입력의 마지막 날 다음 날이 타깃
        assert y[i] == rows[i + SEQ_LEN]["diesel_price"]
        assert y[i] == rows[i + SEQ_LEN - 1]["diesel_price"] + 1


def test_scaler_is_fit_on_training_rows_only():
    rows = validate_rows(make_rows())
    fit_rows = scaler_fit_rows(rows)
    scaler = GasolineScaler().fit(fit_rows)
    X, y = build_sequences(rows, scaler)
    _, y_train, _, y_test = train_test_split(X, y)
    # 학습 타깃은 모두 fit 구간에 있고, 검증 타깃은 하나도 들어가지 않는다
    assert y_train[-1] == fit_rows[-1]["diesel_price"]
    assert min(y_test) > scaler.maximums["diesel_price"]


def test_inverse_transform_restores_won_per_liter():
    rows = validate_rows(make_rows())
    scaler = GasolineScaler().fit(scaler_fit_rows(rows))
    for price in (1500.0, 1537.25, 1700.0):  # 학습 범위 밖 가격도 원/L로 복원
        assert scaler.inverse_price(scaler.scale_price(price)) == pytest.approx(price)


def test_constant_tax_feature_does_not_divide_by_zero():
    rows = validate_rows(make_rows())
    scaler = GasolineScaler().fit(rows)
    changed = {**rows[0], "tax_or_supply_feature": 15.0}
    assert scaler.transform_point(changed)[3] == pytest.approx(-10.0)


@pytest.mark.parametrize("column", ["date", *FEATURE_COLUMNS])
def test_rejects_missing_column(column):
    rows = make_rows(3)
    del rows[1][column]
    with pytest.raises(ValueError, match=column):
        validate_rows(rows)


@pytest.mark.parametrize("column", ["diesel_price", "singapore_diesel_price", "usd_krw"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_rejects_non_positive_or_non_finite(column, value):
    rows = make_rows(3)
    rows[1][column] = value
    with pytest.raises(ValueError):
        validate_rows(rows)


def test_tax_feature_allows_zero_but_not_nan():
    rows = make_rows(3)
    rows[1]["tax_or_supply_feature"] = "0"
    validate_rows(rows)
    rows[1]["tax_or_supply_feature"] = "nan"
    with pytest.raises(ValueError):
        validate_rows(rows)


@pytest.mark.parametrize("change", ["gap", "duplicate", "reversed"])
def test_rejects_broken_date_order(change):
    rows = make_rows(5)
    if change == "gap":
        del rows[2]
    elif change == "duplicate":
        rows[3]["date"] = rows[2]["date"]
    else:
        rows.reverse()
    with pytest.raises(ValueError, match="date"):
        validate_rows(rows)


def test_finetune_split_keeps_targets_disjoint():
    rows = validate_rows(make_rows(80))
    train_rows, val_rows = split_recent_for_finetune(rows, train_days=30, val_days=7)
    scaler = GasolineScaler().fit(rows)
    _, y_train = build_sequences(train_rows, scaler)
    _, y_val = build_sequences(val_rows, scaler)
    assert (len(y_train), len(y_val)) == (30, 7)
    assert y_val == [row["diesel_price"] for row in rows[-7:]]
    assert max(y_train) < min(y_val)  # 시간순: 검증 타깃이 학습 타깃보다 모두 뒤


def test_finetune_split_requires_enough_rows():
    with pytest.raises(ValueError, match="57"):
        split_recent_for_finetune(validate_rows(make_rows(50)))


def test_latest_upload_picks_most_recent_file(tmp_path):
    older, newer = tmp_path / "gasoline_a.csv", tmp_path / "gasoline_b.csv"
    older.write_text("x")
    newer.write_text("x")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    assert storage.latest_upload(str(tmp_path)) == str(newer)


def test_latest_upload_without_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        storage.latest_upload(str(tmp_path))
