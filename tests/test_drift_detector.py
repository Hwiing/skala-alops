"""#15 드리프트 판정·예측 윈도우 테스트: 정상 / 드리프트 / 표본 부족 / 잘못된 값 / 경계값."""

import math
from datetime import date, timedelta

import pytest

from serving_app.monitoring.drift_detector import (
    WINDOW_SIZE,
    compute_rmse,
    evaluate,
    is_drift,
)
from serving_app.monitoring.prediction_window import PredictionWindow


def make_pairs(n, model_err, naive_err, version="1", start=date(2026, 7, 1)):
    """1주차 실제값 1500원 기준. 모델은 ±model_err, naive는 ±naive_err 만큼 빗나간다."""
    pairs = []
    for i in range(n):
        sign = 1 if i % 2 == 0 else -1  # 번갈아 빗나가게 해서 |오차| = RMSE가 되도록
        actual = 1500.0
        pairs.append(
            {
                "date": (start + timedelta(days=i)).isoformat(),
                "predicted": [actual + sign * model_err, 1500.0, 1500.0, 1500.0],
                "actual": [actual, 1500.0, 1500.0, 1500.0],
                "naive": actual - sign * naive_err,
                "model_version": version,
            }
        )
    return pairs


# ---------- compute_rmse ----------


def test_rmse_basic_won_per_liter():
    # 오차 3, -4 → sqrt((9+16)/2) = 3.5355
    pairs = [{"predicted": 1503.0, "actual": 1500.0}, {"predicted": 1496.0, "actual": 1500.0}]
    assert compute_rmse(pairs) == pytest.approx(math.sqrt(12.5))


def test_rmse_empty_is_zero():
    assert compute_rmse([]) == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "1500"])
def test_rmse_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        compute_rmse([{"predicted": bad, "actual": 1500.0}])


# ---------- evaluate: 정상 / 드리프트 / 경계 ----------


def test_ok_when_model_beats_naive():
    result = evaluate(make_pairs(WINDOW_SIZE, model_err=20, naive_err=35))
    assert result["status"] == "ok"
    assert result["week1_rmse"] == 20.0 and result["naive_rmse"] == 35.0
    assert result["n"] == WINDOW_SIZE and result["model_version"] == "1"
    assert result["period"] == ["2026-07-01", "2026-07-28"]
    assert not is_drift(make_pairs(WINDOW_SIZE, 20, 35))


def test_drift_when_model_worse_than_naive():
    result = evaluate(make_pairs(WINDOW_SIZE, model_err=60, naive_err=35))
    assert result["status"] == "drift"
    assert is_drift(make_pairs(WINDOW_SIZE, 60, 35))


def test_tie_with_naive_is_not_drift():
    assert evaluate(make_pairs(WINDOW_SIZE, model_err=35, naive_err=35))["status"] == "ok"


def test_uses_only_latest_window():
    # 오래된 10건은 엉망이어도, 최근 28건이 정상이면 ok
    old = make_pairs(10, 500, 1, start=date(2026, 6, 1))
    recent = make_pairs(WINDOW_SIZE, 20, 35)
    assert evaluate(old + recent)["status"] == "ok"


# ---------- 표본 부족 / 정답 미도착 ----------


def test_insufficient_samples_holds_judgement():
    result = evaluate(make_pairs(WINDOW_SIZE - 1, model_err=500, naive_err=1))
    assert result["status"] == "insufficient_data"
    assert result["n"] == WINDOW_SIZE - 1 and result["week1_rmse"] is None


def test_empty_input_is_insufficient():
    assert evaluate([])["status"] == "insufficient_data"


def test_pending_actuals_are_excluded_and_counted():
    pairs = make_pairs(WINDOW_SIZE, 500, 1)
    for p in pairs[-3:]:
        p["actual"] = [None, None, None, None]  # 아직 정답 미도착
    result = evaluate(pairs)
    assert result["status"] == "insufficient_data"
    assert result["n"] == WINDOW_SIZE - 3 and result["pending"] == 3


# ---------- 모델 버전 ----------


def test_only_latest_model_version_is_judged():
    old_model = make_pairs(WINDOW_SIZE, 500, 1, version="1", start=date(2026, 6, 1))
    new_model = make_pairs(5, 10, 35, version="2", start=date(2026, 7, 1))
    result = evaluate(old_model + new_model)
    # 이전 모델의 큰 오차로 새 모델(v2)을 drift로 판정하면 안 된다
    assert result["status"] == "insufficient_data"
    assert result["model_version"] == "2" and result["n"] == 5


# ---------- 잘못된 값 ----------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(predicted=[float("nan"), 1.0, 1.0, 1.0]),
        lambda p: p.update(actual=[float("inf"), 1.0, 1.0, 1.0]),
        lambda p: p.update(naive=float("nan")),
        lambda p: p.update(predicted=[1500.0]),  # 길이 4가 아님
        lambda p: p.update(actual=None),
    ],
)
def test_invalid_values_are_rejected_not_silently_skipped(mutate):
    pairs = make_pairs(WINDOW_SIZE, 60, 35)
    mutate(pairs[5])
    result = evaluate(pairs)
    assert result["status"] == "invalid" and result["reason"]
    assert not is_drift(pairs)


# ---------- PredictionWindow ----------


def test_window_records_and_evaluates_batch_pairs():
    w = PredictionWindow()
    for p in make_pairs(WINDOW_SIZE, 60, 35, version="3"):
        w.record(p["date"], p["predicted"], p["naive"], "3", actual=p["actual"], source="batch")
    assert len(w) == WINDOW_SIZE
    result = w.evaluate()
    assert result["status"] == "drift" and result["model_version"] == "3"
    assert all(p["source"] == "batch" for p in w.pairs())


def test_window_live_prediction_waits_then_fills_actual():
    w = PredictionWindow()
    w.record("2026-08-01", [1500.0, 1505.0, 1510.0, 1515.0], 1498.0, "3")  # 정답 없이 기록
    assert w.pairs()[0]["actual"] == [None, None, None, None]
    assert w.evaluate()["pending"] == 1
    assert w.fill_actual("2026-08-01", week=1, value=1502.0)
    assert w.pairs()[0]["actual"][0] == 1502.0
    assert not w.fill_actual("2099-01-01", week=1, value=1.0)  # 없는 예측
    with pytest.raises(ValueError):
        w.fill_actual("2026-08-01", week=5, value=1.0)


def test_window_same_date_is_replaced_not_duplicated():
    w = PredictionWindow()
    w.record("2026-08-01", [1.0] * 4, 1.0, "3", actual=[1.0] * 4)
    w.record("2026-08-01", [2.0] * 4, 1.0, "3", actual=[1.0] * 4)  # 배치 재전송
    assert len(w) == 1 and w.pairs()[0]["predicted"] == [2.0] * 4


def test_window_keeps_bounded_size_and_clears():
    w = PredictionWindow(max_records=5)
    for p in make_pairs(8, 1, 1):
        w.record(p["date"], p["predicted"], p["naive"], "1", actual=p["actual"])
    assert len(w) == 5 and w.pairs()[0]["date"] == "2026-07-04"  # 오래된 것부터 버림
    w.clear()
    assert len(w) == 0


def test_window_pairs_returns_copies():
    w = PredictionWindow()
    w.record("2026-08-01", [1.0] * 4, 1.0, "3")
    w.pairs()[0]["actual"][0] = 999.0  # 바깥에서 고쳐도
    assert w.pairs()[0]["actual"][0] is None  # 내부 기록은 그대로
