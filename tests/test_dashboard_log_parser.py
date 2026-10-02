"""대시보드 운영 현황 탭의 aiops.log 파서(serving_app/static/aiops-log.js)를 node로 확인한다 (#56).

node가 없는 환경에서는 건너뛴다. 로그 형식은 retrain_trigger.py가 쓰는 실제 줄을 그대로 쓴다.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PARSER = Path(__file__).parents[1] / "serving_app" / "static" / "aiops-log.js"

LOG = """\
2026-10-02 09:16:23,773 [WARNING] [WARN] drift detected (week1_rmse=56.00 > naive_rmse=4.00, n=28, period=2026-04-30~2026-05-27, model=champion:3)
2026-10-02 09:16:23,773 [INFO] [INFO] retrain data selected: data/uploads/a.csv rows=627 period=2024-06-12~2026-02-28
2026-10-02 09:16:23,773 [INFO] [INFO] retrain triggered (rows=627, data=2024-06-12~2026-02-28)
2026-10-02 09:16:23,774 [INFO] [OK] new_rmse=[30.00, 50.00, 60.00, 70.00] naive=[35.00, 60.00, 80.00, 90.00] - champion promoted: DieselPricePredictor v5
2026-10-02 09:16:23,774 [WARNING] [ALERT] [AIOps INFO] promoted | 2026-10-02T09:16:23+09:00 | 원인: ...
2026-10-02 09:17:00,001 [WARNING] [GATE_FAILED] new_rmse=[60.00, 1.00, 1.00, 1.00] - 1주차 RMSE 60.00 > 50 (Production 유지)
2026-10-02 09:18:00,002 [ERROR] [ERROR] retrain failed: RuntimeError('boom') (Production 유지)
Traceback (most recent call last):
2026-10-02 09:19:00,003 [INFO] [INFO] retrain skipped - cooldown 120s left
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node 없음")
def test_parse_aiops_log_events_and_drift_points():
    script = f"const p = require({json.dumps(str(PARSER))}); process.stdout.write(JSON.stringify(p.parseAiopsLog({json.dumps(LOG)})));"
    out = json.loads(
        subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    )

    events = out["events"]
    # 최신순, [ALERT](같은 결과의 운영자 알림 사본)·재학습 데이터 선택 줄·태그 없는 줄은 뺀다
    assert [e["tag"] for e in events] == ["INFO", "ERROR", "GATE_FAILED", "OK", "INFO", "WARN"]
    assert events[0]["time"] == "2026-10-02 09:19:00"
    assert events[0]["level"] == "info" and events[1]["level"] == "err"
    assert events[2]["level"] == "warn" and events[3]["level"] == "ok"
    assert events[3]["message"].endswith("champion promoted: DieselPricePredictor v5")

    assert out["drift"] == [
        {
            "time": "2026-10-02 09:16:23",
            "week1": 56.0,
            "naive": 4.0,
            "period": "2026-04-30~2026-05-27",
            "model": "champion:3",
        }
    ]
