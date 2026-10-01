"""
대시보드 "운영 현황" 탭용 - 요청 수·응답 시간·성공률 (#56, 발표 항목 2 "품질·응답 시간 운영 목표").

main.py의 미들웨어가 서비스 경로 요청마다 logs/requests.log에 JSON 한 줄을 남기고,
GET /metrics/summary가 기간별로 집계한다. 대시보드 폴링(/health, /logs, /models, /metrics)과
정적 파일은 기록하지 않는다. 배치·업로드는 재학습(수십 초)이 걸릴 수 있어 /predict와 따로 집계한다.

성공률 = 5xx가 아닌 비율. 4xx(입력 검증 실패)는 서버가 정상 응답한 것이므로 성공으로 센다.
"""

import json
import math
import os
import time
from typing import Literal

from fastapi import APIRouter

router = APIRouter(prefix="/metrics")

REQUESTS_LOG = os.path.join("logs", "requests.log")
TRACKED_PATHS = ("/predict", "/predict/batch-test", "/data/upload")
WINDOWS = {"5m": 300, "1h": 3600, "6h": 6 * 3600, "24h": 24 * 3600}


def record(path: str, status: int, duration_ms: float) -> None:
    if path not in TRACKED_PATHS:
        return
    # ponytail: 요청마다 파일 append, 회전 없음. 파일이 커지면 RotatingFileHandler로 교체
    line = {"ts": time.time(), "path": path, "status": status, "duration_ms": round(duration_ms, 1)}
    with open(REQUESTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def _percentile(sorted_values: list[float], p: float) -> float:
    return sorted_values[max(math.ceil(p * len(sorted_values)) - 1, 0)]  # nearest-rank


def _summarize(rows: list[dict]) -> dict:
    if not rows:  # 요청 없음은 0%/0ms가 아니라 값 없음
        return {"requests": 0, "success_rate": None, "p50_ms": None, "p95_ms": None}
    durations = sorted(r["duration_ms"] for r in rows)
    return {
        "requests": len(rows),
        "success_rate": sum(r["status"] < 500 for r in rows) / len(rows),
        "p50_ms": _percentile(durations, 0.5),
        "p95_ms": _percentile(durations, 0.95),
    }


@router.get("/summary")
def summary(window: Literal["5m", "1h", "6h", "24h"] = "5m"):
    since = time.time() - WINDOWS[window]
    rows = []
    if os.path.isfile(REQUESTS_LOG):
        with open(REQUESTS_LOG, encoding="utf-8") as f:
            rows = [r for r in map(json.loads, f) if r["ts"] >= since]
    return {
        "window": window,
        "since": since,
        "paths": {p: _summarize([r for r in rows if r["path"] == p]) for p in TRACKED_PATHS},
    }
