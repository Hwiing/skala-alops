"""
AIOps 2/3 (#16): 운영자 알림 어댑터.

retrain_trigger가 재학습을 시도한 결과(승격·게이트 탈락·실패 등)를 운영자에게 보낸다.

채널 (설정 가능한 어댑터)
    - LogAdapter      : 항상 켜짐. logs/aiops.log에 [ALERT] 한 줄 (채널 장애와 무관한 최소 기록)
    - WebhookAdapter  : AIOPS_ALERT_WEBHOOK_URL이 있으면 켜짐. JSON POST.
                        Slack Incoming Webhook 형식("text")과 구조화 필드를 함께 보낸다.
                        수신 대상 = 그 웹훅이 연결된 채널(예: #aiops-alerts의 운영 담당자).

알림 내용: 시각 · 상태 · 원인(1주차 RMSE vs naive) · RMSE · 임계값(같은 기간 naive RMSE) ·
          모델 버전 · 판정 기간 · 조치 결과(승격 버전/탈락 사유/오류)

중복 억제: 같은 (상태, 모델 버전, 판정 기간) 알림은 AIOPS_ALERT_DEDUP_SECONDS(기본 3600초) 동안
          한 번만 보낸다. 배치가 반복돼도 운영자 채널이 같은 경보로 도배되지 않게 한다.
전송 실패: 어댑터별로 [ERROR] alert send failed 로그를 남기고 나머지 채널은 계속 보낸다.
          알림 실패가 서빙·재학습을 멈추지 않는다.
"""

import json
import logging
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger("aiops")

# 운영자가 알아야 하는 결과만 보낸다. ok·판정 보류·skipped_*는 알림 대상이 아니다.
ALERT_STATUSES = {
    "promoted": "INFO",
    "gate_failed": "WARN",
    "no_production": "ERROR",
    "retrain_failed": "ERROR",
    "insufficient_data": "ERROR",  # 재학습 단계의 데이터 부족 (error 필드가 있을 때만)
}


def build_alert(result: dict) -> dict | None:
    """retrain_trigger 결과 → 알림 dict. 보낼 필요가 없으면 None."""
    status = result.get("status")
    if status not in ALERT_STATUSES:
        return None
    if status == "insufficient_data" and "error" not in result:
        return None  # 판정 단계의 표본 부족(정상적인 보류)은 알리지 않는다
    det = result.get("detection") or {}
    week1, naive = det.get("week1_rmse"), det.get("naive_rmse")
    cause = (
        f"1주차 RMSE {week1:.2f} > 같은 기간 naive RMSE {naive:.2f}"
        if week1 is not None and naive is not None
        else "판정 정보 없음"
    )
    if status == "promoted":
        action = f"새 모델 v{result.get('version')} 승격 (서빙 교체는 batch_test reload 결과 확인)"
    elif status == "gate_failed":
        action = "게이트 탈락, 기존 Production 유지: " + "; ".join(result.get("reasons") or [])
    else:
        action = f"재학습 안 됨, 기존 Production 유지: {result.get('error') or status}"
    return {
        "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "level": ALERT_STATUSES[status],
        "status": status,
        "cause": cause,
        "week1_rmse": week1,
        "threshold_naive_rmse": naive,
        "model_version": det.get("model_version"),
        "period": det.get("period"),
        "new_version": result.get("version"),
        "new_rmse": result.get("rmse"),
        "action": action,
    }


def format_text(alert: dict) -> str:
    period = "~".join(alert["period"]) if alert.get("period") else "-"
    return (
        f"[AIOps {alert['level']}] {alert['status']} | {alert['time']}\n"
        f"원인: {alert['cause']} (모델 {alert.get('model_version') or 'unknown'}, 기간 {period})\n"
        f"조치: {alert['action']}"
    )


class LogAdapter:
    name = "log"

    def send(self, alert: dict) -> None:
        logger.warning("[ALERT] " + format_text(alert).replace("\n", " | "))


class WebhookAdapter:
    name = "webhook"

    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = timeout

    def send(self, alert: dict) -> None:
        body = json.dumps({"text": format_text(alert), "alert": alert}, ensure_ascii=False)
        req = urllib.request.Request(
            self.url,
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # 4xx·5xx는 HTTPError
            if resp.status >= 300:
                raise RuntimeError(f"webhook HTTP {resp.status}")


class OperatorNotifier:
    def __init__(self, adapters: list, dedup_seconds: float = 3600.0, clock=time.monotonic):
        self.adapters = adapters
        self.dedup_seconds = dedup_seconds
        self._clock = clock
        self._sent: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def notify(self, result: dict) -> dict:
        """retrain_trigger.notifier로 등록되는 함수. 채널별 전송 결과를 돌려준다."""
        alert = build_alert(result)
        if alert is None:
            return {"sent": False, "reason": "not_alertable"}
        key = (alert["status"], alert["model_version"], tuple(alert["period"] or ()))
        now = self._clock()
        with self._lock:
            last = self._sent.get(key)
            if last is not None and now - last < self.dedup_seconds:
                logger.info(f"[INFO] alert suppressed (duplicate within {self.dedup_seconds:.0f}s)")
                return {"sent": False, "reason": "duplicate"}
            self._sent[key] = now
        channels = {}
        for adapter in self.adapters:
            try:
                adapter.send(alert)
                channels[adapter.name] = "ok"
            except Exception as exc:
                logger.error(f"[ERROR] alert send failed via {adapter.name}: {exc!r}")
                channels[adapter.name] = f"failed: {exc!r}"
        return {"sent": True, "channels": channels, "alert": alert}


def from_env() -> OperatorNotifier:
    adapters: list = [LogAdapter()]
    url = os.getenv("AIOPS_ALERT_WEBHOOK_URL")
    if url:
        adapters.append(WebhookAdapter(url, float(os.getenv("AIOPS_ALERT_TIMEOUT", "5"))))
    return OperatorNotifier(adapters, float(os.getenv("AIOPS_ALERT_DEDUP_SECONDS", "3600")))
