"""경유 v2 드리프트 계약: 최근 28개 기준일의 1주 RMSE > 같은 기간 naive RMSE.

배포 게이트 50원과 탐지 기준은 별개다. 계산 구현은 AIOps 담당 TODO이며 성공으로 표시하지 않는다.
"""

from data.contracts import PAIR_WINDOW

WINDOW_SIZE = PAIR_WINDOW


def compute_rmse(recent_predictions: list[dict], *, naive: bool = False) -> float:
    """BatchPair 목록의 1주차 RMSE.

    TODO: actual[0]과 predicted[0]의 오차, naive=True면 actual[0]과 naive의 오차를
    sqrt(mean(error**2))로 계산한다. 빈 목록은 0.0으로 반환한다.
    목록의 date는 예측 기준일이며 같은 기준일의 정답과 비교한다.
    """
    raise NotImplementedError("compute_rmse: 경유 1주차·naive RMSE 구현 필요")


def is_drift(recent_predictions: list[dict]) -> bool:
    if len(recent_predictions) < WINDOW_SIZE:
        return False  # 호출자는 insufficient_data로 구분한다.
    window = recent_predictions[-WINDOW_SIZE:]
    return compute_rmse(window) > compute_rmse(window, naive=True)
