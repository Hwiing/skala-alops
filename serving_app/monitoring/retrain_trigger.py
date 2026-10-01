"""경유 v2 탐지 → 재학습 어댑터. 실제 연결은 AIOps 담당 TODO.

입력은 BatchPair 목록이고 출력은 serving_app.schemas.DriftCheck 계약을 따른다.
승격 뒤 서빙 교체·기록 초기화는 라우터가 담당한다.
"""

import logging

from data.contracts import PAIR_WINDOW
from serving_app.monitoring.drift_detector import is_drift

logger = logging.getLogger("aiops")


def check_and_trigger(recent_predictions: list[dict]) -> dict:
    if len(recent_predictions) < PAIR_WINDOW:
        return {"status": "insufficient_data", "reasons": [f"판정 짝 {PAIR_WINDOW}개 필요"]}
    if not is_drift(recent_predictions):
        return {"status": "ok"}

    logger.warning("[WARN] drift detected - triggering retrain")
    # TODO: 중복 기준일·동시/중복 트리거 방지, 운영자 알림, 실행 실패 복구.
    # 저장된 전체 데이터에서 최근 data.contracts.FINETUNE_MIN_ROWS(627)행을 확보하고
    # serving_app.diesel_registry.fine_tune(rows)를 호출한다. 최신 업로드 175행만 쓰지 않는다.
    # 부족하면 insufficient_data, warm start 불가면 no_production, 예외는 retrain_failed.
    # FineTuneResult의 promoted/gate_failed/no_production 상태와 4주 지표를 그대로 전달하고
    # 기존 모델 탐지 지표 drift_rmse/drift_naive_rmse와 reasons를 함께 기록한다.
    # Registry 승격을 실제 서빙 교체 성공으로 표시하지 않는다.
    raise NotImplementedError("TODO: 경유 627행 fine-tuning·운영자 알림·중복 억제 연결")
