"""
AIOps 3/3 (#17): 드리프트 감지 → 로그·운영자 알림 → fine-tuning → 게이트 → 결과 반환.

흐름 (contracts.md v2)
    1) drift_detector.evaluate()로 판정. ok·insufficient_data·invalid면 여기서 끝.
    2) drift면 중복·동시 실행·쿨다운을 확인하고 막힌 경우 skipped_*를 반환한다.
    3) 저장된 전체 경유 데이터의 최근 FINETUNE_MIN_ROWS(627)행으로 diesel_registry.fine_tune() 호출.
       업로드 배치(175행)만으로는 fine-tuning(학습 365일 + 검증 90일)에 부족하다.
    4) fine_tune 결과(status = promoted / gate_failed / no_production)를 그대로 돌려준다.
       promoted면 version이 반드시 들어 있다 → 서빙(#34)이 champion이 그 버전인지 확인한 뒤 교체한다.

역할 분담 (#14·#34)
    모델 캐시 교체(reload_model)와 예측 기록 초기화는 서빙이 한다. 여기서는 하지 않는다.
    게이트 탈락·예외 때는 승격이 일어나지 않으므로 기존 Production이 그대로 서빙된다.

반복 재학습 방지
    - 같은 판정(모델 버전 + 판정 기간)으로는 한 번만 재학습한다 (skipped_duplicate).
      게이트 탈락·교체 실패 후 서빙이 기록을 유지해도, 같은 데이터로 다시 학습하지 않는다.
    - 승격하지 못한 재학습 뒤에는 RETRAIN_COOLDOWN_SECONDS 동안 새 재학습을 막는다 (skipped_cooldown).
    - 재학습이 이미 돌고 있으면 기다리지 않고 바로 돌려준다 (skipped_in_progress).
      fine-tuning은 수십 초 걸리므로 요청을 붙잡아 두면 배치가 줄줄이 막힌다.

로그 (logs/aiops.log, "aiops" 로거 - main.py에서 파일 연결)
    [WARN] drift detected ... → [INFO] retrain triggered ... → [OK] ... | [GATE_FAILED] ... | [ERROR] ...
"""

import logging
import os
import threading
import time
from collections.abc import Callable

from serving_app.monitoring.drift_detector import evaluate

logger = logging.getLogger("aiops")

MODEL_NAME = "DieselPricePredictor"
DATA_CSV = os.getenv("DIESEL_DATA_CSV", "data/processed/diesel_features_2008_spliced.csv")
RETRAIN_COOLDOWN_SECONDS = float(os.getenv("RETRAIN_COOLDOWN_SECONDS", "600"))

# 운영자 알림 훅 (#16에서 웹훅·메일 어댑터를 연결). event dict 하나를 받는 함수.
notifier: Callable[[dict], None] | None = None

_retrain_lock = threading.Lock()
_attempted: set[tuple] = set()  # 이미 재학습을 시도한 (모델 버전, 판정 시작일, 판정 종료일)
_last_failure_at: float | None = None  # 마지막으로 승격하지 못한 재학습의 종료 시각


def _load_recent_rows() -> list[dict]:
    """저장된 전체 경유 데이터에서 fine-tuning에 필요한 최근 행만 자른다."""
    from data.diesel_features import FINETUNE_MIN_ROWS, load_diesel_rows

    return load_diesel_rows(DATA_CSV)[-FINETUNE_MIN_ROWS:]


def _fine_tune(rows: list[dict]) -> dict:
    # mlflow·tensorflow는 무거우므로 실제로 재학습할 때만 import 한다
    from serving_app.diesel_registry import fine_tune

    return fine_tune(rows)


def _notify(event: dict) -> None:
    """운영자 알림. 알림이 실패해도 재학습 결과는 그대로 돌려준다 (실패는 로그로 남김)."""
    if notifier is None:
        return
    try:
        notifier(event)
    except Exception as exc:  # 알림 채널 장애가 서빙·재학습을 멈추면 안 된다
        logger.error(f"[ERROR] operator notify failed: {exc!r} (event={event.get('status')})")


def _fmt(values) -> str:
    if values is None:
        return "-"
    if isinstance(values, (list, tuple)):
        return "[" + ", ".join(f"{v:.2f}" for v in values) + "]"
    return f"{values:.2f}"


def _finish(result: dict, detection: dict) -> dict:
    """반환 직전 공통 처리: 판정 근거를 붙이고, 승격 실패면 쿨다운 시작, 알림 전송."""
    global _last_failure_at
    result["detection"] = detection
    if not result.get("promoted"):
        _last_failure_at = time.monotonic()
    _notify(result)
    return result


def check_and_trigger(recent_predictions: list[dict]) -> dict:
    """batch_test(#34)가 호출. 서빙은 promoted·version만 읽고 나머지는 응답에 그대로 싣는다."""
    detection = evaluate(recent_predictions)
    status = detection["status"]

    if status == "invalid":
        logger.warning(f"[WARN] drift check rejected - invalid input: {detection['reason']}")
        return {"status": "invalid_input", "promoted": False, "detection": detection}
    if status != "drift":  # ok 또는 insufficient_data (판정 보류)
        return {"status": status, "promoted": False, "detection": detection}

    key = (detection["model_version"], *detection["period"])
    period = f"{detection['period'][0]}~{detection['period'][1]}"
    logger.warning(
        f"[WARN] drift detected (week1_rmse={detection['week1_rmse']:.2f} > "
        f"naive_rmse={detection['naive_rmse']:.2f}, n={detection['n']}, period={period}, "
        f"model={detection['model_version'] or 'unknown'})"
    )

    if key in _attempted:
        logger.info(f"[INFO] retrain skipped - already attempted for period={period}")
        return {"status": "skipped_duplicate", "promoted": False, "detection": detection}
    if _last_failure_at is not None:
        remaining = RETRAIN_COOLDOWN_SECONDS - (time.monotonic() - _last_failure_at)
        if remaining > 0:
            logger.info(f"[INFO] retrain skipped - cooldown {remaining:.0f}s left")
            return {
                "status": "skipped_cooldown",
                "promoted": False,
                "retry_after_seconds": round(remaining),
                "detection": detection,
            }
    if not _retrain_lock.acquire(blocking=False):
        logger.info("[INFO] retrain skipped - another retrain is in progress")
        return {"status": "skipped_in_progress", "promoted": False, "detection": detection}

    try:
        _attempted.add(key)  # 성공·실패와 관계없이 같은 판정으로는 다시 시도하지 않는다
        try:
            rows = _load_recent_rows()
            logger.info(
                f"[INFO] retrain triggered (rows={len(rows)}, "
                f"data={rows[0]['date']}~{rows[-1]['date']})"
            )
            result = _fine_tune(rows)
        except (ValueError, FileNotFoundError) as exc:
            if isinstance(exc, FileNotFoundError) or "insufficient_data" in str(exc):
                logger.error(
                    f"[ERROR] retrain skipped - insufficient data: {exc} (Production 유지)"
                )
                return _finish(
                    {"status": "insufficient_data", "promoted": False, "error": str(exc)},
                    detection,
                )
            raise
    except Exception as exc:
        logger.error(f"[ERROR] retrain failed: {exc!r} (Production 유지)")
        return _finish(
            {"status": "retrain_failed", "promoted": False, "error": repr(exc)}, detection
        )
    finally:
        _retrain_lock.release()

    if result.get("promoted"):
        if result.get("version") is None:  # 계약 위반: 서빙이 교체를 확인할 수 없다
            logger.error("[ERROR] fine_tune promoted without version - 교체 확인 불가")
            result = {
                **result,
                "status": "retrain_failed",
                "promoted": False,
                "error": "promoted result has no version",
            }
        else:
            logger.info(
                f"[OK] new_rmse={_fmt(result.get('rmse'))} naive={_fmt(result.get('naive_rmse'))}"
                f" - champion promoted: {MODEL_NAME} v{result['version']}"
            )
    elif result.get("status") == "no_production":
        logger.error("[ERROR] fine-tune skipped - Production 없음 (base 모델 먼저 배포 필요)")
    else:
        logger.warning(
            f"[GATE_FAILED] new_rmse={_fmt(result.get('rmse'))} - "
            f"{'; '.join(result.get('reasons') or [])} (Production 유지)"
        )
    return _finish(dict(result), detection)


def reset_state() -> None:
    """테스트·운영자 수동 재시도용: 중복·쿨다운 기록을 지운다."""
    global _last_failure_at
    _attempted.clear()
    _last_failure_at = None
