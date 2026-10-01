"""
AIOps 3/3 (#17): 드리프트 감지 → 로그·운영자 알림 → fine-tuning → 게이트 → DriftCheck 반환.

반환은 serving_app.schemas.DriftCheck 공개 계약(docs/contracts.md "AIOps 상태와 서빙 교체")을 따른다.
    status: ok | insufficient_data | promoted | gate_failed | no_production | retrain_failed
    drift_rmse / drift_naive_rmse : 기존 서빙 모델의 탐지 지표(1주차, 스칼라)
    rmse[4] / naive_rmse[4] ...   : 재학습 후 독립 검증 지표 (fine_tune 결과 그대로)
    reasons                       : 판정·조치 이유
탐지 내부 상태(drift·invalid)와 부가 통계(기간·모델 버전·pending)는 API에 싣지 않고 로그·알림에만 쓴다.

흐름
    1) drift_detector.evaluate()  ok·insufficient_data는 여기서 끝. invalid는 판정 불가 → insufficient_data.
    2) drift → [WARN] 로그 → 중복·쿨다운·동시 실행 확인
    3) 저장된 전체 경유 데이터의 최근 FINETUNE_MIN_ROWS(627)행으로 diesel_registry.fine_tune()
       (업로드 배치 175행만으로는 학습 365일 + 검증 90일에 부족)
    4) fine_tune 결과(promoted/gate_failed/no_production)에 탐지 지표를 붙여 반환
       promoted면 version 필수 → 서빙(#34)이 champion이 그 버전인지 확인한 뒤 교체·기록 초기화.
       게이트 탈락·예외는 승격이 없으므로 기존 Production이 그대로 서빙된다.

반복 재학습 방지
    - 같은 판정(모델 버전 + 판정 기간)으로는 한 번만 재학습하고, 다시 들어오면 그때 결과를 재사용한다.
      서빙이 교체에 실패해 기록을 유지한 경우, 재사용된 promoted 결과로 서빙이 교체를 다시 시도할 수 있다.
    - 승격하지 못한 재학습 뒤에는 RETRAIN_COOLDOWN_SECONDS 동안 새 재학습을 막는다.
    - 재학습이 이미 돌고 있으면 기다리지 않고 바로 돌려준다 (fine-tuning은 수십 초).
    쿨다운·진행 중은 재학습을 하지 않은 상태라 retrain_failed + 이유로 알린다 (정상 ok로 숨기지 않음).

로그 (logs/aiops.log, "aiops" 로거)
    [WARN] drift detected → [INFO] retrain triggered → [OK] | [GATE_FAILED] | [ERROR]
"""

import logging
import os
import threading
import time
from collections.abc import Callable

from data.contracts import FINETUNE_MIN_ROWS, MODEL_NAME
from serving_app.monitoring import notifier as operator_notifier
from serving_app.monitoring.drift_detector import WINDOW_SIZE, evaluate

logger = logging.getLogger("aiops")

DATA_CSV = os.getenv("DIESEL_DATA_CSV", "data/processed/diesel_features_2008_spliced.csv")
RETRAIN_COOLDOWN_SECONDS = float(os.getenv("RETRAIN_COOLDOWN_SECONDS", "600"))

# 운영자 알림 훅: 결과 dict + "detection"을 받는 함수.
# 기본은 환경변수로 구성한 OperatorNotifier(로그 + 선택적 웹훅, #16).
notifier: Callable[[dict], object] | None = operator_notifier.from_env().notify

_retrain_lock = threading.Lock()
_results: dict[tuple, dict] = {}  # (모델 버전, 판정 시작일, 종료일) → 그 판정의 재학습 결과
_last_failure_at: float | None = None  # 마지막으로 승격하지 못한 재학습의 종료 시각


def _load_recent_rows() -> list[dict]:
    """저장된 전체 경유 데이터에서 fine-tuning에 필요한 최근 행만 자른다."""
    from data.diesel_features import load_diesel_rows

    return load_diesel_rows(DATA_CSV)[-FINETUNE_MIN_ROWS:]


def _fine_tune(rows: list[dict]) -> dict:
    # mlflow·tensorflow는 무거우므로 실제로 재학습할 때만 import 한다
    from serving_app.diesel_registry import fine_tune

    return fine_tune(rows)


def _notify(public: dict, detection: dict) -> None:
    """운영자 알림. 알림이 실패해도 재학습 결과는 그대로 돌려준다 (실패는 로그로 남김)."""
    if notifier is None:
        return
    try:
        notifier({**public, "detection": detection})
    except Exception as exc:  # 알림 채널 장애가 서빙·재학습을 멈추면 안 된다
        logger.error(f"[ERROR] operator notify failed: {exc!r} (status={public.get('status')})")


def _fmt(values) -> str:
    if values is None:
        return "-"
    if isinstance(values, (list, tuple)):
        return "[" + ", ".join(f"{v:.2f}" for v in values) + "]"
    return f"{values:.2f}"


def _detection_fields(detection: dict) -> dict:
    out = {}
    if detection.get("week1_rmse") is not None:
        out["drift_rmse"] = detection["week1_rmse"]
        out["drift_naive_rmse"] = detection["naive_rmse"]
    return out


def _detection_reason(detection: dict) -> str:
    period = "~".join(detection["period"])
    return (
        f"탐지: 1주차 RMSE {detection['week1_rmse']:.2f} > 같은 기간 naive RMSE "
        f"{detection['naive_rmse']:.2f} (기준일 {detection['n']}개 {period}, "
        f"모델 {detection['model_version'] or 'unknown'})"
    )


def _not_retrained(detection: dict, reason: str) -> dict:
    """재학습을 시작하지 않은 drift 결과 (쿨다운·진행 중). 기존 모델 유지."""
    return {
        "status": "retrain_failed",
        "promoted": False,
        **_detection_fields(detection),
        "reasons": [_detection_reason(detection), reason],
    }


def _finish(public: dict, detection: dict, key: tuple) -> dict:
    """재학습을 시도한 결과의 공통 처리: 기록(중복 재사용), 실패면 쿨다운 시작, 알림."""
    global _last_failure_at
    _results[key] = public
    if not public.get("promoted"):
        _last_failure_at = time.monotonic()
    _notify(public, detection)
    return public


def check_and_trigger(recent_predictions: list[dict]) -> dict:
    """batch_test가 호출. 반환은 DriftCheck 계약 (서빙은 promoted·version으로 교체를 판단)."""
    detection = evaluate(recent_predictions)
    status = detection["status"]

    if status == "invalid":
        logger.warning(f"[WARN] drift check rejected - invalid input: {detection['reason']}")
        return {
            "status": "insufficient_data",
            "reasons": [f"판정 불가(잘못된 입력): {detection['reason']}"],
        }
    if status == "insufficient_data":
        return {
            "status": "insufficient_data",
            "reasons": [
                f"정답이 확보된 기준일 {detection['n']}/{WINDOW_SIZE}개"
                + (f", 정답 대기 {detection['pending']}개" if detection["pending"] else "")
            ],
        }
    if status == "ok":
        return {"status": "ok", **_detection_fields(detection)}

    key = (detection["model_version"], *detection["period"])
    period = "~".join(detection["period"])
    logger.warning(
        f"[WARN] drift detected (week1_rmse={detection['week1_rmse']:.2f} > "
        f"naive_rmse={detection['naive_rmse']:.2f}, n={detection['n']}, period={period}, "
        f"model={detection['model_version'] or 'unknown'})"
    )

    if key in _results:
        logger.info(f"[INFO] retrain skipped - already attempted for period={period}")
        previous = _results[key]
        return {
            **previous,
            "reasons": [
                *previous.get("reasons", []),
                "같은 판정은 재학습하지 않음(이전 결과 재사용)",
            ],
        }
    if _last_failure_at is not None:
        remaining = RETRAIN_COOLDOWN_SECONDS - (time.monotonic() - _last_failure_at)
        if remaining > 0:
            logger.info(f"[INFO] retrain skipped - cooldown {remaining:.0f}s left")
            return _not_retrained(
                detection,
                f"재학습 대기: 직전 실패 후 쿨다운 {remaining:.0f}초 남음, 기존 모델 유지",
            )
    if not _retrain_lock.acquire(blocking=False):
        logger.info("[INFO] retrain skipped - another retrain is in progress")
        return _not_retrained(detection, "다른 재학습이 진행 중, 기존 모델 유지")

    reason = _detection_reason(detection)
    try:
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
                public = {
                    "status": "insufficient_data",
                    "promoted": False,
                    **_detection_fields(detection),
                    "reasons": [reason, f"재학습 데이터 부족: {exc}"],
                }
                return _finish(public, detection, key)
            raise
    except Exception as exc:
        logger.error(f"[ERROR] retrain failed: {exc!r} (Production 유지)")
        public = {
            "status": "retrain_failed",
            "promoted": False,
            **_detection_fields(detection),
            "reasons": [reason, f"재학습 실패: {exc!r}"],
        }
        return _finish(public, detection, key)
    finally:
        _retrain_lock.release()

    public = {
        **result,
        **_detection_fields(detection),
        "reasons": [reason, *(result.get("reasons") or [])],
    }
    if public.get("promoted") and not public.get("version"):
        logger.error("[ERROR] fine_tune promoted without version - 교체 확인 불가")
        public = {
            "status": "retrain_failed",
            "promoted": False,
            **_detection_fields(detection),
            "reasons": [reason, "승격 결과에 version이 없어 서빙 교체를 확인할 수 없음"],
        }
    elif public.get("promoted"):
        logger.info(
            f"[OK] new_rmse={_fmt(public.get('rmse'))} naive={_fmt(public.get('naive_rmse'))}"
            f" - champion promoted: {MODEL_NAME} v{public['version']}"
        )
    elif public.get("status") == "no_production":
        logger.error("[ERROR] fine-tune skipped - Production 없음 (base 모델 먼저 배포 필요)")
    else:
        logger.warning(
            f"[GATE_FAILED] new_rmse={_fmt(public.get('rmse'))} - "
            f"{'; '.join(result.get('reasons') or [])} (Production 유지)"
        )
    return _finish(public, detection, key)


def reset_state() -> None:
    """테스트·운영자 수동 재시도용: 중복·쿨다운 기록을 지운다."""
    global _last_failure_at
    _results.clear()
    _last_failure_at = None
