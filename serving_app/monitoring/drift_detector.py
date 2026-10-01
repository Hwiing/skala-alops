"""
AIOps 1/3 (#15): 예측·실제값 짝으로 성능 저하(드리프트)를 판정한다.

판정 기준 (docs/contracts.md v2):
    정답이 확보된 최근 WINDOW_SIZE(28)건에서
        1주차 모델 RMSE  >  같은 기간 naive RMSE   →  drift
    - 1주차 정답은 예측 7일 뒤에야 확보되므로, 정답이 없는 짝은 판정에서 뺀다(보류).
    - naive = 마지막 입력일 가격을 그대로 쓰는 "무변화 예측". 고정 임계값(예: 10원) 대신
      같은 기간의 naive와 비교하면, 시장 전체가 출렁여 모두가 틀리는 시기와
      "우리 모델만 망가진" 시기를 구분할 수 있다.
    - 동점(같음)은 drift가 아니다. 게이트(diesel_gate)는 RMSE < naive 여야 통과이므로,
      naive와 같다는 것은 "합격선 위에 있다가 내려온 것"이 아니라 경계값일 뿐이다.

짝(pair) 형식 - batch_test·prediction_window가 넘기는 dict:
    {"date": "2026-08-01",            # 마지막 입력일(base_date)
     "predicted": [p1, p2, p3, p4],   # 1~4주 평균 예측 (원/L)
     "actual":    [a1, a2, a3, a4],   # 실제 평균. 아직 안 왔으면 None
     "naive": 1541.0,                 # 마지막 입력일 가격
     "model_version": "3"}            # 이 예측을 만든 모델 버전 (없으면 None)

반환 status:
    "ok"           - 판정 완료, 정상
    "drift"        - 판정 완료, 성능 저하 → retrain_trigger가 재학습 여부를 결정
    "insufficient_data" - 정답이 확보된 짝이 WINDOW_SIZE 미만 → 판정 보류
    "invalid"      - 잘못된 값(NaN·inf·형식 오류) → 판정 거부 (조용히 넘어가지 않는다)
"""

import math

WINDOW_SIZE = 28  # 정답이 확보된 최근 28건 (4주). contracts.md v2 드리프트 제안
HORIZONS = 4  # 1~4주 예측
WEEK1 = 0  # 드리프트 판정은 1주차(가장 빨리 정답이 오는 주차)로만 한다


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compute_rmse(pairs: list[dict]) -> float:
    """pairs: [{"predicted": float, "actual": float}, ...] → RMSE (원/L).

    RMSE = sqrt( mean( (actual - predicted) ** 2 ) )
    - 빈 리스트는 0.0 (오차를 계산할 짝이 없음).
    - NaN·inf가 섞이면 ValueError. NaN은 비교 연산이 항상 False라
      "rmse > 기준"이 False → 드리프트를 조용히 놓치는 원인이 된다.
    """
    if not pairs:
        return 0.0
    for p in pairs:
        if not (_finite_number(p.get("predicted")) and _finite_number(p.get("actual"))):
            raise ValueError(f"비유한 값 또는 숫자가 아닌 값: {p}")
    squared = [(p["actual"] - p["predicted"]) ** 2 for p in pairs]
    return math.sqrt(sum(squared) / len(squared))


def _validate_pair(pair: dict) -> str | None:
    """잘못된 짝이면 이유 문자열, 정상이면 None."""
    predicted, actual = pair.get("predicted"), pair.get("actual")
    if not isinstance(predicted, (list, tuple)) or len(predicted) != HORIZONS:
        return f"predicted는 길이 {HORIZONS} 리스트여야 합니다: {pair.get('date')}"
    if not isinstance(actual, (list, tuple)) or len(actual) != HORIZONS:
        return f"actual은 길이 {HORIZONS} 리스트여야 합니다(미도착은 None): {pair.get('date')}"
    if not all(_finite_number(v) for v in predicted):
        return f"predicted에 비유한 값: {pair.get('date')}"
    if not all(v is None or _finite_number(v) for v in actual):
        return f"actual에 비유한 값: {pair.get('date')}"
    if not _finite_number(pair.get("naive")):
        return f"naive에 비유한 값: {pair.get('date')}"
    return None


def evaluate(pairs: list[dict]) -> dict:
    """짝 목록으로 드리프트를 판정해 상태 dict를 돌려준다 (batch_test·retrain_trigger용 계약).

    1) 잘못된 값이 하나라도 있으면 invalid  - 일부만 버리고 판정하면 결과를 믿을 수 없다.
    2) 가장 최근 모델 버전의 짝만 쓴다    - 교체 전 모델의 오차로 새 모델을 탓하지 않도록.
    3) 1주차 정답이 온 짝만 쓴다          - 미도착 짝은 pending으로 세고 보류.
    4) 최근 WINDOW_SIZE건으로 모델 RMSE vs naive RMSE.
    """
    for pair in pairs:
        reason = _validate_pair(pair)
        if reason:
            return {"status": "invalid", "reason": reason}

    model_version = pairs[-1].get("model_version") if pairs else None
    same_model = [p for p in pairs if p.get("model_version") == model_version]
    matured = [p for p in same_model if p["actual"][WEEK1] is not None]
    pending = len(same_model) - len(matured)

    base = {"model_version": model_version, "window_size": WINDOW_SIZE, "pending": pending}
    if len(matured) < WINDOW_SIZE:
        return {
            "status": "insufficient_data",
            "n": len(matured),
            "week1_rmse": None,
            "naive_rmse": None,
            "period": None,
            **base,
        }

    window = matured[-WINDOW_SIZE:]
    week1_rmse = compute_rmse(
        [{"predicted": p["predicted"][WEEK1], "actual": p["actual"][WEEK1]} for p in window]
    )
    naive_rmse = compute_rmse(
        [{"predicted": p["naive"], "actual": p["actual"][WEEK1]} for p in window]
    )
    return {
        "status": "drift" if week1_rmse > naive_rmse else "ok",
        "n": len(window),
        "week1_rmse": round(week1_rmse, 2),
        "naive_rmse": round(naive_rmse, 2),
        "period": [window[0].get("date"), window[-1].get("date")],
        **base,
    }


def is_drift(pairs: list[dict]) -> bool:
    """retrain_trigger 호환용 단축 함수. 판정 보류·잘못된 값은 drift가 아니다(재학습 안 함)."""
    return evaluate(pairs)["status"] == "drift"
