"""
경유 배포 게이트 (contracts.md v2). MLflow·TensorFlow 없이 판단만 하는 순수 함수라 CI에서 테스트한다.

통과 조건 (모두 만족):
1) 1주차 RMSE ≤ 50원/L — 화물 안전운임제 재고시 기준(3개월 평균 ±50원)을 업무 허용 오차로 사용
2) 1~4주 모두 RMSE < naive RMSE — 유가 예측 연구의 표준 비교(무변화 예측 대비 우위)
3) Production이 있으면 1~4주 평균 RMSE ≤ Production 1~4주 평균 RMSE (같은 검증 구간).
   한 주차만 보면 쇼크 한 번에 판정이 뒤집혀서 평균으로 비교한다(evidence/10).
비유한 값(NaN·Infinity)이 하나라도 있으면 실패.
"""

from math import isfinite

WEEK1_RMSE_MAX = 50.0


def check_gate(
    rmse: list[float], naive_rmse: list[float], production_rmse: list[float] | None = None
) -> dict:
    """반환: {"passed": bool, "reasons": [실패 이유...]}. 이유가 없으면 통과."""
    values = [*rmse, *naive_rmse, *(production_rmse or [])]
    if not values or not all(isinstance(v, (int, float)) and isfinite(v) for v in values):
        return {"passed": False, "reasons": ["비유한 RMSE(NaN/Infinity) 또는 빈 값"]}
    if len(rmse) != len(naive_rmse):
        return {"passed": False, "reasons": ["모델과 naive의 주차 수가 다름"]}
    reasons = []
    if rmse[0] > WEEK1_RMSE_MAX:
        reasons.append(f"1주차 RMSE {rmse[0]:.2f} > {WEEK1_RMSE_MAX:.0f}")
    for k, (m, n) in enumerate(zip(rmse, naive_rmse), start=1):
        if not m < n:
            reasons.append(f"{k}주차 RMSE {m:.2f} >= naive {n:.2f}")
    if production_rmse:
        if len(production_rmse) != len(rmse):
            return {"passed": False, "reasons": ["모델과 Production의 주차 수가 다름"]}
        mean, prod_mean = sum(rmse) / len(rmse), sum(production_rmse) / len(production_rmse)
        if mean > prod_mean:
            reasons.append(f"1~4주 평균 RMSE {mean:.2f} > Production {prod_mean:.2f}")
    return {"passed": not reasons, "reasons": reasons}
