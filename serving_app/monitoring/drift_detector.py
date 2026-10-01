"""
AIOps 1/3 (#15): 경유 v2 드리프트 판정 (docs/contracts.md "AIOps 상태와 서빙 교체").

    정답이 확보된 최근 PAIR_WINDOW(28)개 기준일에서
        1주차 모델 RMSE  >  같은 기간 naive RMSE   →  drift

    - naive = 기준일 가격을 1주 뒤에도 그대로 쓰는 "무변화 예측". 고정 임계값(예: 10원) 대신
      같은 기간 naive와 비교하면, 시장 급변으로 모두가 틀리는 시기와 "우리 모델만 망가진" 시기를
      구분할 수 있다. 배포 게이트 50원과 탐지 기준은 별개다.
    - 동점은 drift가 아니다 (게이트 통과 조건이 RMSE < naive이므로 경계값).

입력 짝 (HTTP BatchPair + 내부 기록용 선택 필드)
    {"date": "2026-04-30",                 # 입력 기준일
     "predicted": [p1, p2, p3, p4],         # 1~4주 평균 예측 (원/L)
     "actual":    [a1, a2, a3, a4],         # 실제 평균. 실운영 기록에서 미도착이면 None
     "naive": 1619.0,                       # 기준일 가격
     "model_version": "champion:3"}         # 내부 기록에만 있음(prediction_window). 없으면 None

evaluate()의 status는 탐지 내부 상태다 (API DriftCheck가 아님 - retrain_trigger가 변환).
    "ok" / "drift"         판정 완료
    "insufficient_data"    정답 확보 짝이 28개 미만 → 판정 보류 (정상으로 간주하지 않음)
    "invalid"              잘못된 값 → 판정 거부 (일부만 버리고 판정하지 않는다)
"""

import math

from data.contracts import HORIZONS, PAIR_WINDOW

WINDOW_SIZE = PAIR_WINDOW
WEEK1 = 0  # 드리프트 판정은 1주차(가장 빨리 정답이 오는 주차)로만 한다


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compute_rmse(pairs: list[dict], *, naive: bool = False) -> float:
    """짝 목록의 1주차 RMSE (원/L). naive=True면 같은 짝의 naive 예측 RMSE.

    RMSE = sqrt( mean( (actual[0] - 예측) ** 2 ) ),  예측 = predicted[0] 또는 naive
    - 빈 목록은 0.0.
    - NaN·inf·None이 섞이면 ValueError. NaN은 비교가 항상 False라 "rmse > naive"가 False →
      드리프트를 조용히 놓치는 원인이 된다.
    """
    if not pairs:
        return 0.0
    squared = []
    for p in pairs:
        guess = p.get("naive") if naive else (p.get("predicted") or [None])[WEEK1]
        actual = (p.get("actual") or [None])[WEEK1]
        if not (_finite_number(guess) and _finite_number(actual)):
            raise ValueError(f"1주차 값이 유한한 숫자가 아닙니다: {p.get('date')}")
        squared.append((actual - guess) ** 2)
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
    if not pair.get("date"):
        return "date가 없습니다"
    return None


def evaluate(pairs: list[dict]) -> dict:
    """짝 목록으로 드리프트를 판정한다.

    1) 잘못된 값이 하나라도 있으면 invalid.
    2) 가장 최근 짝의 model_version과 같은 버전의 짝만 쓴다 - 교체 전 모델 오차로 새 모델을 탓하지 않는다.
    3) 같은 기준일이 여러 번 들어오면 마지막 것 하나만 센다 - 배치 재전송으로 재집계하지 않는다.
    4) 1주차 정답이 온 짝만 쓰고, 미도착은 pending으로 센다.
    5) 최근 28개로 모델 RMSE vs naive RMSE.
    """
    for pair in pairs:
        reason = _validate_pair(pair)
        if reason:
            return {"status": "invalid", "reason": reason}

    model_version = pairs[-1].get("model_version") if pairs else None
    latest_by_date: dict[str, dict] = {}
    for p in pairs:
        if p.get("model_version") == model_version:
            latest_by_date.pop(str(p["date"]), None)  # 다시 넣어 순서를 최신으로
            latest_by_date[str(p["date"])] = p
    same_model = sorted(latest_by_date.values(), key=lambda p: str(p["date"]))
    matured = [p for p in same_model if p["actual"][WEEK1] is not None]

    base = {
        "model_version": model_version,
        "window_size": WINDOW_SIZE,
        "pending": len(same_model) - len(matured),
    }
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
    week1_rmse = compute_rmse(window)
    naive_rmse = compute_rmse(window, naive=True)
    return {
        "status": "drift" if week1_rmse > naive_rmse else "ok",
        "n": len(window),
        "week1_rmse": round(week1_rmse, 2),
        "naive_rmse": round(naive_rmse, 2),
        "period": [str(window[0]["date"]), str(window[-1]["date"])],
        **base,
    }


def is_drift(pairs: list[dict]) -> bool:
    """판정 보류·잘못된 값은 drift가 아니다 (재학습하지 않음)."""
    return evaluate(pairs)["status"] == "drift"
