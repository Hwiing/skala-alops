from datetime import date, timedelta

import pytest

from data.diesel_features import (
    FEATURES,
    GAP_WINDOW,
    HORIZONS,
    SEQ_LEN,
    DailyFrame,
    FeatureScaler,
    build_windows,
)
from data.diesel_policy import CapSchedule, TaxSchedule, pass_ratio


def make_rows(n=200, start="2015-01-01", price=lambda i: 1400 + i):
    """유류세 변경이 없는 2015년 구간. 가격이 하루 1원씩 올라 타깃 정렬을 값으로 확인할 수 있다."""
    return [
        {
            "date": (date.fromisoformat(start) + timedelta(days=i)).isoformat(),
            "diesel_price": float(price(i)),
            "singapore_diesel_price": 70.0 + i / 100,
            "usd_krw": 1100.0,
            "tax_or_supply_feature": 0.0,
        }
        for i in range(n)
    ]


def test_tax_total_matches_law_amounts():
    taxes = TaxSchedule()
    # 2008-04-15 교통세 335원 × (1 + 교육세 15% + 주행세 27%), 2009-05-21 375원 × 1.41
    assert taxes.tax("2008-04-15") == pytest.approx(475.70)
    assert taxes.tax("2009-05-21") == pytest.approx(528.75)
    with pytest.raises(ValueError):
        taxes.tax("2008-01-01")


def test_pass_ratio_and_lagged_retail_tax():
    assert pass_ratio(-1) == 0.0
    assert pass_ratio(0) == pytest.approx(0.33)
    assert pass_ratio(14) == 1.0
    taxes = TaxSchedule()
    before, after = taxes.tax("2018-11-05"), taxes.tax("2018-11-06")
    assert taxes.retail_tax("2018-11-06") == pytest.approx(before + (after - before) * 0.33)
    assert taxes.retail_tax("2018-11-20") == pytest.approx(after)


def test_tax_change_known_only_after_announcement():
    taxes = TaxSchedule()
    before, after = taxes.tax("2026-03-26"), taxes.tax("2026-03-27")
    # 2026-03-27 인하(25%)는 3/26 발표 → 3/25 시점 예측은 기존 세율이 이어진다고 본다
    assert taxes.retail_tax("2026-04-20", as_of="2026-03-25") == pytest.approx(before)
    assert taxes.retail_tax("2026-04-20", as_of="2026-03-26") == pytest.approx(after)
    assert taxes.retail_tax("2026-04-20") == pytest.approx(after)  # as_of 없으면 실제 시행 기준


def test_cap_known_only_after_announcement():
    caps = CapSchedule()
    assert caps.cap("2026-03-12") is None
    # 2차 상한(1923원, 3/27 시행)은 3/26 발표 → 3/25 시점엔 1차(1713원)가 이어진다고 본다
    assert caps.known_cap("2026-03-30", as_of="2026-03-25") == 1713
    assert caps.known_cap("2026-03-30", as_of="2026-03-26") == 1923
    assert caps.retail_change("2026-03-27", as_of="2026-03-25") == 0.0
    assert caps.retail_change("2026-03-27", as_of="2026-03-26") == pytest.approx(210 * 0.33)


def test_features_use_only_past():
    rows = make_rows()
    i = 150
    base = DailyFrame(rows).features(i)
    changed = make_rows(price=lambda j: 1400 + j + (500 if j > i else 0))
    assert DailyFrame(changed).features(i) == base
    assert len(base) == len(FEATURES)
    assert DailyFrame(rows).features(GAP_WINDOW - 2) is None


def test_targets_align_with_weekly_windows():
    frame = DailyFrame(make_rows())
    # 세금 변화 없음 + 하루 1원 상승 → k주 평균(i+7k-6 ~ i+7k)과 오늘의 차이 = 7k-3
    assert frame.targets(100) == pytest.approx([4, 11, 18, 25])
    assert frame.actual(100) == pytest.approx([1500 + 4, 1500 + 11, 1500 + 18, 1500 + 25])
    assert frame.targets(len(frame.price) - 7 * HORIZONS) is None


def test_policy_without_cap_adds_change_to_price():
    frame = DailyFrame(make_rows())
    assert frame.apply_policy(100, [1, 2, 3, 4]) == pytest.approx([1501, 1502, 1503, 1504])


def test_windows_and_scaler_clip():
    frame = DailyFrame(make_rows())
    X, Y, kept = build_windows(frame, range(len(frame.price)))
    assert kept[0] == GAP_WINDOW - 1 + SEQ_LEN - 1
    assert len(X[0]) == SEQ_LEN and len(X[0][0]) == len(FEATURES) and len(Y[0]) == HORIZONS
    scaler = FeatureScaler().fit(X[:20], Y[:20])
    extreme = [[v * 1000 for v in step] for step in X[0]]
    hi = scaler.transform([scaler.hi])[0]
    assert all(a <= b + 1e-9 for step in scaler.transform(extreme) for a, b in zip(step, hi))


def test_finetune_split_has_no_target_overlap():
    from data.diesel_features import FINETUNE_MIN_ROWS, split_finetune

    frame = DailyFrame(make_rows(n=FINETUNE_MIN_ROWS))
    train_idx, val_idx = split_finetune(frame)
    assert len(train_idx) == 365 and len(val_idx) == 90
    # 학습 마지막 날의 4주 평균 정답 구간이 검증 첫날 이전에 끝난다
    assert train_idx[-1] + 7 * HORIZONS < val_idx[0]
    assert frame.targets(val_idx[-1]) is not None


def test_finetune_split_fails_explicitly_when_short():
    from data.diesel_features import FINETUNE_MIN_ROWS, split_finetune

    with pytest.raises(ValueError, match="insufficient_data"):
        split_finetune(DailyFrame(make_rows(n=FINETUNE_MIN_ROWS - 1)))
