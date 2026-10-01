"""#16 운영자 알림: 알림 대상 선별 · 내용 · 웹훅 수신/실패 · 중복 억제 · retrain_trigger 연결."""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from serving_app.monitoring import notifier as nt

DETECTION = {
    "status": "drift",
    "week1_rmse": 76.0,
    "naive_rmse": 4.0,
    "model_version": "champion:3",
    "period": ["2026-04-30", "2026-05-27"],
}


def result(status="promoted", **extra):
    base = {"status": status, "promoted": status == "promoted", "detection": DETECTION}
    if status == "promoted":
        base.update(version="4", rmse=[20.0, 30.0, 40.0, 50.0])
    return {**base, **extra}


class Recorder:
    name = "recorder"

    def __init__(self):
        self.alerts = []

    def send(self, alert):
        self.alerts.append(alert)


class Broken:
    name = "broken"

    def send(self, alert):
        raise ConnectionError("channel down")


@pytest.fixture
def webhook():
    """실제 HTTP로 받는 모의 웹훅. state["fail"]=True면 500."""
    state = {"received": [], "fail": False}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if state["fail"]:
                self.send_response(500)
                self.end_headers()
                return
            state["received"].append(json.loads(body))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


# ---------- 알림 대상 ----------


@pytest.mark.parametrize(
    "status",
    ["ok", "skipped_duplicate", "skipped_cooldown", "skipped_in_progress", "invalid_input"],
)
def test_routine_statuses_are_not_alerted(status):
    assert nt.build_alert({"status": status, "detection": DETECTION}) is None


def test_detection_hold_is_not_alerted_but_training_data_shortage_is():
    hold = {"status": "insufficient_data", "detection": {"status": "insufficient_data"}}
    assert nt.build_alert(hold) is None  # 판정 보류(짝 부족)는 정상적인 대기
    shortage = result("insufficient_data", reasons=["탐지: ...", "재학습 데이터 부족: 627행 필요"])
    alert = nt.build_alert(shortage)
    assert alert["level"] == "ERROR" and "627행" in alert["action"]
    assert "탐지:" not in alert["action"]  # 원인은 cause 필드에 따로


def test_alert_contains_time_cause_rmse_threshold_version():
    alert = nt.build_alert(result())
    assert alert["time"] and alert["status"] == "promoted"
    assert alert["cause"] == "1주차 RMSE 76.00 > 같은 기간 naive RMSE 4.00"
    assert alert["week1_rmse"] == 76.0 and alert["threshold_naive_rmse"] == 4.0
    assert alert["model_version"] == "champion:3" and alert["new_version"] == "4"
    assert alert["period"] == ["2026-04-30", "2026-05-27"]


def test_gate_failed_alert_explains_reasons():
    alert = nt.build_alert(result("gate_failed", reasons=["1주차 RMSE 60.00 > 50"]))
    assert alert["level"] == "WARN" and "Production 유지" in alert["action"]
    assert "60.00 > 50" in alert["action"]


# ---------- 전송 ----------


def test_webhook_receives_slack_compatible_payload(webhook):
    n = nt.OperatorNotifier([nt.WebhookAdapter(webhook["url"], timeout=2)])
    out = n.notify(result())
    assert out["sent"] and out["channels"] == {"webhook": "ok"}
    payload = webhook["received"][0]
    assert payload["text"].startswith("[AIOps INFO] promoted")
    assert "naive RMSE 4.00" in payload["text"] and payload["alert"]["new_version"] == "4"


def test_webhook_failure_is_logged_and_other_channels_still_sent(webhook, caplog):
    webhook["fail"] = True
    rec = Recorder()
    n = nt.OperatorNotifier([nt.WebhookAdapter(webhook["url"], timeout=2), rec])
    with caplog.at_level(logging.INFO, logger="aiops"):
        out = n.notify(result())
    assert out["channels"]["webhook"].startswith("failed")
    assert out["channels"]["recorder"] == "ok" and len(rec.alerts) == 1
    assert "[ERROR] alert send failed via webhook" in caplog.text


def test_unreachable_webhook_does_not_raise(caplog):
    n = nt.OperatorNotifier([nt.WebhookAdapter("http://127.0.0.1:9", timeout=1), Broken()])
    with caplog.at_level(logging.INFO, logger="aiops"):
        out = n.notify(result())
    assert out["sent"] and all(v.startswith("failed") for v in out["channels"].values())
    assert caplog.text.count("[ERROR] alert send failed") == 2


def test_log_adapter_writes_alert_line(caplog):
    with caplog.at_level(logging.INFO, logger="aiops"):
        nt.OperatorNotifier([nt.LogAdapter()]).notify(
            result("retrain_failed", reasons=["재학습 실패: boom"])
        )
    assert "[ALERT] [AIOps ERROR] retrain_failed" in caplog.text


# ---------- 중복 억제 ----------


def test_same_alert_is_suppressed_within_window():
    now = [1000.0]
    rec = Recorder()
    n = nt.OperatorNotifier([rec], dedup_seconds=3600, clock=lambda: now[0])
    assert n.notify(result("gate_failed"))["sent"]
    assert n.notify(result("gate_failed")) == {"sent": False, "reason": "duplicate"}
    assert n.notify(result("retrain_failed", reasons=["x"]))["sent"]  # 다른 상태는 따로 보냄
    now[0] += 3601
    assert n.notify(result("gate_failed"))["sent"]  # 억제 시간이 지나면 다시 보냄
    assert len(rec.alerts) == 3


def test_from_env_adds_webhook_only_when_configured(monkeypatch):
    monkeypatch.delenv("AIOPS_ALERT_WEBHOOK_URL", raising=False)
    assert [a.name for a in nt.from_env().adapters] == ["log"]
    monkeypatch.setenv("AIOPS_ALERT_WEBHOOK_URL", "http://example.invalid/hook")
    monkeypatch.setenv("AIOPS_ALERT_DEDUP_SECONDS", "60")
    n = nt.from_env()
    assert [a.name for a in n.adapters] == ["log", "webhook"] and n.dedup_seconds == 60


# ---------- retrain_trigger 연결 ----------


def test_retrain_trigger_sends_alert_through_notifier(monkeypatch):
    from serving_app.monitoring import retrain_trigger as rt
    from tests.test_drift_detector import make_pairs

    rec = Recorder()
    rt.reset_state()
    monkeypatch.setattr(rt, "notifier", nt.OperatorNotifier([rec]).notify)
    monkeypatch.setattr(rt, "_load_recent_rows", lambda: [{"date": "a"}, {"date": "b"}])
    monkeypatch.setattr(
        rt,
        "_fine_tune",
        lambda rows: {
            "promoted": True,
            "status": "promoted",
            "version": "9",
            "rmse": [1.0] * 4,
            "naive_rmse": [2.0] * 4,
        },
    )
    rt.check_and_trigger(make_pairs(28, model_err=60, naive_err=35, version="3"))
    rt.reset_state()
    assert rec.alerts[0]["status"] == "promoted" and rec.alerts[0]["new_version"] == "9"
    assert rec.alerts[0]["model_version"] == "3"
