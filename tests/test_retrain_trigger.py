"""#17 재학습 트리거: 정상·보류·승격·게이트 탈락·데이터 부족·학습 실패·중복·쿨다운·동시 실행·알림."""

import logging
from datetime import date

import pytest

from serving_app.monitoring import retrain_trigger as rt
from serving_app.monitoring.drift_detector import WINDOW_SIZE
from tests.test_drift_detector import make_pairs

DRIFT = dict(model_err=60, naive_err=35)  # 모델이 naive보다 나쁨 → drift
NORMAL = dict(model_err=20, naive_err=35)


@pytest.fixture
def env(monkeypatch):
    """fine_tune·데이터 로드를 가짜로 바꾸고, 호출 기록과 반환값을 조작할 수 있게 한다."""
    rt.reset_state()
    state = {
        "result": {
            "promoted": True,
            "status": "promoted",
            "version": "5",
            "rmse": [30.0, 50.0, 60.0, 70.0],
            "naive_rmse": [35.0, 60.0, 80.0, 90.0],
        },
        "error": None,
        "calls": [],
        "events": [],
    }

    def fake_rows():
        return [{"date": "2024-01-01"}, {"date": "2026-08-31"}]

    def fake_fine_tune(rows):
        state["calls"].append(rows)
        if state["error"]:
            raise state["error"]
        return dict(state["result"])

    monkeypatch.setattr(rt, "_load_recent_rows", fake_rows)
    monkeypatch.setattr(rt, "_fine_tune", fake_fine_tune)
    monkeypatch.setattr(rt, "notifier", state["events"].append)
    monkeypatch.setattr(rt, "RETRAIN_COOLDOWN_SECONDS", 600)
    yield state
    rt.reset_state()


def drift_pairs(start=date(2026, 7, 1), version="3"):
    return make_pairs(WINDOW_SIZE, version=version, start=start, **DRIFT)


# ---------- 판정만 하고 끝나는 경우 ----------


def test_ok_does_not_retrain(env):
    result = rt.check_and_trigger(make_pairs(WINDOW_SIZE, **NORMAL))
    assert result["status"] == "ok" and result["promoted"] is False
    assert result["detection"]["week1_rmse"] == 20.0
    assert env["calls"] == [] and env["events"] == []


def test_insufficient_samples_holds(env):
    result = rt.check_and_trigger(make_pairs(WINDOW_SIZE - 1, **DRIFT))
    assert result["status"] == "insufficient_data" and env["calls"] == []


def test_invalid_input_is_rejected(env):
    pairs = drift_pairs()
    pairs[0]["naive"] = float("nan")
    result = rt.check_and_trigger(pairs)
    assert result["status"] == "invalid_input" and env["calls"] == []


# ---------- 재학습 결과 ----------


def test_promoted_returns_version_and_logs_in_order(env, caplog):
    with caplog.at_level(logging.INFO, logger="aiops"):
        result = rt.check_and_trigger(drift_pairs())
    assert result["status"] == "promoted" and result["promoted"] is True
    assert result["version"] == "5"  # 서빙이 champion 확인에 쓰는 값
    assert result["detection"]["status"] == "drift"
    assert len(env["calls"]) == 1
    messages = [r.getMessage() for r in caplog.records]
    assert [m.split("]")[0] + "]" for m in messages] == ["[WARN]", "[INFO]", "[OK]"]
    assert "v5" in messages[-1]
    assert env["events"][-1]["status"] == "promoted"


def test_gate_failed_keeps_production(env, caplog):
    env["result"] = {
        "promoted": False,
        "status": "gate_failed",
        "rmse": [60.0, 1, 1, 1],
        "naive_rmse": [35.0, 1, 1, 1],
        "reasons": ["1주차 RMSE 60.00 > 50"],
    }
    with caplog.at_level(logging.INFO, logger="aiops"):
        result = rt.check_and_trigger(drift_pairs())
    assert result["status"] == "gate_failed" and result["promoted"] is False
    assert "version" not in result
    assert "[GATE_FAILED]" in caplog.records[-1].getMessage()


def test_no_production(env):
    env["result"] = {"promoted": False, "status": "no_production", "rmse": None}
    assert rt.check_and_trigger(drift_pairs())["status"] == "no_production"


@pytest.mark.parametrize(
    "error",
    [ValueError("insufficient_data: 최소 627행"), FileNotFoundError("diesel csv 없음")],
    ids=["too_few_rows", "missing_csv"],
)
def test_insufficient_training_data(env, error):
    env["error"] = error
    result = rt.check_and_trigger(drift_pairs())
    assert result["status"] == "insufficient_data" and result["promoted"] is False
    assert result["error"]


def test_training_exception_is_reported_not_raised(env, caplog):
    env["error"] = RuntimeError("mlflow down")
    with caplog.at_level(logging.INFO, logger="aiops"):
        result = rt.check_and_trigger(drift_pairs())
    assert result["status"] == "retrain_failed" and result["promoted"] is False
    assert "mlflow down" in result["error"]
    assert caplog.records[-1].getMessage().startswith("[ERROR]")


def test_promoted_without_version_is_not_reported_as_promoted(env):
    env["result"] = {"promoted": True, "status": "promoted", "rmse": [1.0] * 4}
    result = rt.check_and_trigger(drift_pairs())
    assert result["promoted"] is False and result["status"] == "retrain_failed"


# ---------- 반복·동시 재학습 방지 ----------


def test_same_detection_is_retrained_only_once(env):
    env["result"] = {"promoted": False, "status": "gate_failed", "rmse": [60.0] * 4}
    rt.check_and_trigger(drift_pairs())
    second = rt.check_and_trigger(drift_pairs())  # 서빙이 기록을 유지 → 같은 판정이 다시 들어옴
    assert second["status"] == "skipped_duplicate"
    assert len(env["calls"]) == 1


def test_cooldown_after_failure_then_retry(env, monkeypatch):
    env["result"] = {"promoted": False, "status": "gate_failed", "rmse": [60.0] * 4}
    rt.check_and_trigger(drift_pairs())
    later = rt.check_and_trigger(drift_pairs(start=date(2026, 7, 2)))  # 새 판정 구간
    assert later["status"] == "skipped_cooldown" and later["retry_after_seconds"] > 0
    assert len(env["calls"]) == 1

    monkeypatch.setattr(rt, "RETRAIN_COOLDOWN_SECONDS", 0)  # 쿨다운이 지난 것으로
    assert rt.check_and_trigger(drift_pairs(start=date(2026, 7, 2)))["status"] == "gate_failed"
    assert len(env["calls"]) == 2


def test_no_cooldown_after_promotion(env):
    rt.check_and_trigger(drift_pairs(version="3"))
    # 승격 후 새 모델(v5)의 판정은 바로 재학습할 수 있어야 한다
    result = rt.check_and_trigger(drift_pairs(version="5"))
    assert result["status"] == "promoted" and len(env["calls"]) == 2


def test_concurrent_retrain_is_skipped(env):
    assert rt._retrain_lock.acquire(blocking=False)  # 다른 요청이 재학습 중인 상황
    try:
        result = rt.check_and_trigger(drift_pairs())
    finally:
        rt._retrain_lock.release()
    assert result["status"] == "skipped_in_progress" and env["calls"] == []


# ---------- 운영자 알림 ----------


def test_notifier_failure_does_not_break_result(env, monkeypatch, caplog):
    def broken(event):
        raise ConnectionError("webhook timeout")

    monkeypatch.setattr(rt, "notifier", broken)
    with caplog.at_level(logging.INFO, logger="aiops"):
        result = rt.check_and_trigger(drift_pairs())
    assert result["status"] == "promoted"
    assert "operator notify failed" in caplog.records[-1].getMessage()
