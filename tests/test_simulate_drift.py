"""#16 시나리오 생성기·배치 전송 (서버 없이 검증)."""

import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from scripts import simulate_drift as sd

COLUMNS = {"date", "diesel_price", "singapore_diesel_price", "usd_krw", "tax_or_supply_feature"}


@pytest.mark.parametrize("name", list(sd.SCENARIOS))
def test_rows_follow_batch_contract(name):
    rows = sd.generate_rows(name)
    assert len(rows) == sd.BATCH_N == 175
    assert all(set(r) == COLUMNS for r in rows)
    days = [date.fromisoformat(r["date"]) for r in rows]
    assert all((b - a).days == 1 for a, b in zip(days, days[1:]))  # 하루 간격, 중복·누락 없음
    assert all(r["diesel_price"] > 0 and r["usd_krw"] > 0 for r in rows)


def test_same_seed_is_reproducible_and_seed_changes_noise():
    assert sd.generate_rows("oil_shock", seed=42) == sd.generate_rows("oil_shock", seed=42)
    assert sd.generate_rows("normal", seed=42) != sd.generate_rows("normal", seed=7)


def test_scenarios_differ_from_control_only_after_shock():
    control = sd.generate_rows("normal")
    for name in ["oil_shock", "fx_shock", "tax_policy", "supply_disruption"]:
        rows = sd.generate_rows(name)
        assert rows[: sd.SHOCK_DAY] == control[: sd.SHOCK_DAY]  # 충격 전에는 대조군과 똑같다
        assert rows[sd.SHOCK_DAY :] != control[sd.SHOCK_DAY :]


def test_shock_sizes():
    control = sd.generate_rows("normal")
    oil, fx, tax, supply = (
        sd.generate_rows(n) for n in ["oil_shock", "fx_shock", "tax_policy", "supply_disruption"]
    )
    last = -1
    assert oil[last]["singapore_diesel_price"] / control[last][
        "singapore_diesel_price"
    ] == pytest.approx(1.35, rel=1e-3)
    assert fx[last]["usd_krw"] / control[last]["usd_krw"] == pytest.approx(1.10, rel=1e-3)
    assert (
        tax[last]["tax_or_supply_feature"] == 0.0 and control[last]["tax_or_supply_feature"] == 10.0
    )
    jump = supply[sd.SHOCK_DAY]["diesel_price"] - control[sd.SHOCK_DAY]["diesel_price"]
    assert jump == pytest.approx(120, abs=0.01)
    assert (
        supply[sd.SHOCK_DAY]["singapore_diesel_price"]
        == control[sd.SHOCK_DAY]["singapore_diesel_price"]
    )
    assert "소매 +120원" in sd.describe(supply, control)
    assert sd.describe(control, control) == "충격 없음 (대조군)"


def test_unknown_scenario():
    with pytest.raises(ValueError):
        sd.generate_rows("meteor")


@pytest.fixture
def server():
    state = {"status": 200, "body": {"drift_check": {"status": "ok"}}, "received": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["received"].append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(state["body"]).encode())

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_port}/predict/batch-test"
    yield state
    srv.shutdown()


def test_send_batch_posts_rows_and_returns_drift_check(server):
    rows = sd.generate_rows("normal")
    out = sd.send_batch(rows, "normal", server["url"], timeout=5)
    assert out == {"label": "normal", "http_status": 200, "drift_check": {"status": "ok"}}
    assert server["received"][0] == {"rows": rows}


@pytest.mark.parametrize("status", [422, 501, 503])
def test_send_batch_reports_http_errors(server, status):
    server["status"], server["body"] = status, {"detail": "TODO"}
    out = sd.send_batch(sd.generate_rows("normal"), "normal", server["url"], timeout=5)
    assert out["http_status"] == status and "TODO" in out["error"]


def test_send_batch_reports_connection_failure():
    out = sd.send_batch([], "x", "http://127.0.0.1:9/predict/batch-test", timeout=1)
    assert out["http_status"] is None and "연결 실패" in out["error"]


def test_summary_table():
    result = {
        "label": "oil_shock",
        "http_status": 200,
        "drift_check": {
            "status": "promoted",
            "version": "4",
            "reload": {"reloaded": True, "version": "champion:4"},
            "detection": {"status": "drift", "week1_rmse": 76.0, "naive_rmse": 4.0},
        },
    }
    s = sd.summarize(result, "소매 +280원")
    assert s["detection"] == "drift" and s["action"] == "promoted" and s["reload"] == "champion:4"
    table = sd.to_markdown([s], "meta")
    assert (
        "| oil_shock | 소매 +280원 | 200 | drift | 76.00 | 4.00 | promoted | 4 | champion:4 |"
        in table
    )
