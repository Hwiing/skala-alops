"""#17 재학습 트리거: 모든 반환이 DriftCheck 공개 계약(6개 상태)을 지키는지와
정상·보류·승격·게이트 탈락·데이터 부족·학습 실패·중복·쿨다운·동시 실행·알림을 검증한다."""

import logging
from datetime import date

import pytest

from serving_app.monitoring import retrain_trigger as rt
from serving_app.monitoring.drift_detector import WINDOW_SIZE
from serving_app.schemas import DriftCheck
from tests.test_drift_detector import make_pairs

DRIFT = dict(model_err=60, naive_err=35)  # 모델이 naive보다 나쁨 → drift
NORMAL = dict(model_err=20, naive_err=35)
PROMOTED = {
    "status": "promoted",
    "promoted": True,
    "version": "5",
    "passed": True,
    "rmse": [30.0, 50.0, 60.0, 70.0],
    "naive_rmse": [35.0, 60.0, 80.0, 90.0],
    "production_rmse": [31.0, 52.0, 61.0, 72.0],
    "production_before": "3",
    "run_id": "run-1",
    "reasons": [],
}
GATE_FAILED = {
    **PROMOTED,
    "status": "gate_failed",
    "promoted": False,
    "version": None,
    "passed": False,
    "rmse": [60.0, 1.0, 1.0, 1.0],
    "reasons": ["1주차 RMSE 60.00 > 50"],
}


@pytest.fixture
def env(monkeypatch):
    """fine_tune·데이터 로드를 가짜로 바꾸고, 호출 기록과 반환값을 조작할 수 있게 한다."""
    rt.reset_state()
    state = {"result": dict(PROMOTED), "error": None, "calls": [], "events": []}

    def fake_fine_tune(rows):
        state["calls"].append(rows)
        if state["error"]:
            raise state["error"]
        return dict(state["result"])

    monkeypatch.setattr(
        rt, "_load_recent_rows", lambda: [{"date": "2024-06-12"}, {"date": "2026-02-28"}]
    )
    monkeypatch.setattr(rt, "_fine_tune", fake_fine_tune)
    monkeypatch.setattr(rt, "notifier", state["events"].append)
    monkeypatch.setattr(rt, "RETRAIN_COOLDOWN_SECONDS", 600)
    yield state
    rt.reset_state()


def check(pairs) -> dict:
    """check_and_trigger 결과가 서빙이 검증하는 DriftCheck 계약을 통과해야 한다."""
    result = rt.check_and_trigger(pairs)
    DriftCheck.model_validate(result)
    return result


def drift_pairs(start=date(2026, 7, 1), version="champion:3"):
    return make_pairs(WINDOW_SIZE, version=version, start=start, **DRIFT)


# ---------- 판정만 하고 끝나는 경우 ----------


def test_ok_reports_detection_metrics_without_retrain(env):
    result = check(make_pairs(WINDOW_SIZE, **NORMAL))
    assert result == {"status": "ok", "drift_rmse": 20.0, "drift_naive_rmse": 35.0}
    assert env["calls"] == [] and env["events"] == []


def test_insufficient_pairs_hold_judgement(env):
    result = check(make_pairs(WINDOW_SIZE - 1, **DRIFT))
    assert result["status"] == "insufficient_data" and "27/28" in result["reasons"][0]
    assert env["calls"] == []


def test_invalid_input_is_not_reported_as_normal(env):
    pairs = drift_pairs()
    pairs[0]["naive"] = float("nan")
    result = check(pairs)
    assert result["status"] == "insufficient_data" and "잘못된 입력" in result["reasons"][0]
    assert env["calls"] == []


# ---------- 재학습 결과 ----------


def test_promoted_keeps_finetune_metrics_and_adds_detection(env, caplog):
    with caplog.at_level(logging.INFO, logger="aiops"):
        result = check(drift_pairs())
    assert result["status"] == "promoted" and result["promoted"] is True
    assert result["version"] == "5"  # 서빙이 champion 확인에 쓰는 값
    assert result["rmse"] == PROMOTED["rmse"]  # 재학습 4주 지표
    assert (result["drift_rmse"], result["drift_naive_rmse"]) == (60.0, 35.0)  # 탐지 1주 지표
    assert result["reasons"][0].startswith("탐지: 1주차 RMSE 60.00 > 같은 기간 naive RMSE 35.00")
    assert len(env["calls"]) == 1
    messages = [r.getMessage() for r in caplog.records]
    assert [m.split("]")[0] + "]" for m in messages] == ["[WARN]", "[INFO]", "[OK]"]
    assert "v5" in messages[-1]
    event = env["events"][-1]
    assert event["status"] == "promoted" and event["detection"]["model_version"] == "champion:3"


def test_gate_failed_keeps_production(env, caplog):
    env["result"] = dict(GATE_FAILED)
    with caplog.at_level(logging.INFO, logger="aiops"):
        result = check(drift_pairs())
    assert result["status"] == "gate_failed" and result["promoted"] is False
    assert "version" not in result or result["version"] is None
    assert "1주차 RMSE 60.00 > 50" in result["reasons"]
    assert "[GATE_FAILED]" in caplog.records[-1].getMessage()


def test_no_production(env):
    env["result"] = {"status": "no_production", "promoted": False, "reasons": ["Production 없음"]}
    result = check(drift_pairs())
    assert result["status"] == "no_production" and "Production 없음" in result["reasons"]


@pytest.mark.parametrize(
    "error",
    [ValueError("insufficient_data: 최소 627행"), FileNotFoundError("diesel csv 없음")],
    ids=["too_few_rows", "missing_csv"],
)
def test_insufficient_training_data(env, error):
    env["error"] = error
    result = check(drift_pairs())
    assert result["status"] == "insufficient_data" and result["promoted"] is False
    assert "재학습 데이터 부족" in result["reasons"][-1]
    assert result["drift_rmse"] == 60.0


def test_training_exception_is_reported_not_raised(env, caplog):
    env["error"] = RuntimeError("mlflow down")
    with caplog.at_level(logging.INFO, logger="aiops"):
        result = check(drift_pairs())
    assert result["status"] == "retrain_failed" and result["promoted"] is False
    assert "mlflow down" in result["reasons"][-1]
    assert caplog.records[-1].getMessage().startswith("[ERROR]")


def test_promoted_without_version_is_not_reported_as_promoted(env):
    env["result"] = {**PROMOTED, "version": None}
    result = check(drift_pairs())
    assert result["promoted"] is False and result["status"] == "retrain_failed"


# ---------- 반복·동시 재학습 방지 ----------


def test_same_detection_reuses_previous_result(env):
    env["result"] = dict(GATE_FAILED)
    first = check(drift_pairs())
    second = check(drift_pairs())  # 서빙이 기록을 유지 → 같은 판정이 다시 들어옴
    assert len(env["calls"]) == 1
    assert second["status"] == first["status"] == "gate_failed"
    assert second["reasons"][-1] == "같은 판정은 재학습하지 않음(이전 결과 재사용)"


def test_promoted_result_is_reused_so_serving_can_retry_reload(env):
    check(drift_pairs())
    again = check(drift_pairs())  # 교체 실패로 기록이 남아 같은 판정
    assert again["promoted"] is True and again["version"] == "5"
    assert len(env["calls"]) == 1


def test_cooldown_after_failure_then_retry(env, monkeypatch):
    env["result"] = dict(GATE_FAILED)
    check(drift_pairs())
    later = check(drift_pairs(start=date(2026, 7, 2)))  # 새 판정 구간
    assert later["status"] == "retrain_failed" and "쿨다운" in later["reasons"][-1]
    assert len(env["calls"]) == 1

    monkeypatch.setattr(rt, "RETRAIN_COOLDOWN_SECONDS", 0)  # 쿨다운이 지난 것으로
    assert check(drift_pairs(start=date(2026, 7, 2)))["status"] == "gate_failed"
    assert len(env["calls"]) == 2


def test_no_cooldown_after_promotion(env):
    check(drift_pairs(version="champion:3"))
    result = check(drift_pairs(version="champion:5"))  # 승격 후 새 모델의 판정
    assert result["status"] == "promoted" and len(env["calls"]) == 2


def test_concurrent_retrain_is_skipped(env):
    assert rt._retrain_lock.acquire(blocking=False)  # 다른 요청이 재학습 중인 상황
    try:
        result = check(drift_pairs())
    finally:
        rt._retrain_lock.release()
    assert result["status"] == "retrain_failed" and "진행 중" in result["reasons"][-1]
    assert env["calls"] == []


# ---------- 알림 ----------


def test_notifier_failure_does_not_break_result(env, monkeypatch, caplog):
    def broken(event):
        raise ConnectionError("webhook timeout")

    monkeypatch.setattr(rt, "notifier", broken)
    with caplog.at_level(logging.INFO, logger="aiops"):
        result = check(drift_pairs())
    assert result["status"] == "promoted"
    assert "operator notify failed" in caplog.records[-1].getMessage()


def test_main_contract_insufficient_on_empty():
    assert DriftCheck.model_validate(rt.check_and_trigger([])).status == "insufficient_data"


# ---------- 서빙 통합: /predict/batch-test → 실제 판정·트리거 → reload ----------


@pytest.fixture
def served(env, monkeypatch):
    from fastapi.testclient import TestClient

    from serving_app import model_loader
    from serving_app.main import app
    from serving_app.monitoring.prediction_window import PredictionWindow
    from serving_app.routers import predict as predict_router
    from tests.test_drift_detector import _ShiftModel
    from tests.test_scaffold import make_rows

    reloads = []
    reload_ok = {"value": True}

    def fake_reload(version):
        reloads.append(version)
        if reload_ok["value"]:
            return {"reloaded": True, "version": f"champion:{version}"}
        return {"reloaded": False, "version": "champion:3", "error": "load failed"}

    monkeypatch.setattr(predict_router, "recent_predictions", PredictionWindow())
    monkeypatch.setattr(model_loader, "_model_cache", _ShiftModel(60.0, "champion:3"))
    monkeypatch.setattr(model_loader, "reload_model", fake_reload)
    client = TestClient(app)

    def post():
        return client.post("/predict/batch-test", json={"rows": make_rows(175)})

    return post, predict_router, reloads, reload_ok


def test_http_drift_promotes_reloads_and_clears_window(served, env):
    post, router, reloads, _ = served
    response = post()
    assert response.status_code == 200
    check_ = response.json()["drift_check"]
    assert check_["status"] == "promoted" and check_["reload"]["reloaded"] is True
    assert check_["drift_rmse"] == 56.0 and check_["drift_naive_rmse"] == 4.0
    assert reloads == ["5"] and len(router.recent_predictions) == 0


def test_http_reload_failure_keeps_window_and_retries_without_retraining(served, env):
    post, router, reloads, reload_ok = served
    reload_ok["value"] = False
    first = post().json()["drift_check"]
    assert first["reload"]["reloaded"] is False and len(router.recent_predictions) == 28
    reload_ok["value"] = True
    second = post().json()["drift_check"]  # 같은 판정 → 이전 promoted 재사용 → 교체 재시도
    assert second["reload"]["reloaded"] is True and reloads == ["5", "5"]
    assert len(env["calls"]) == 1  # 재학습은 한 번만
