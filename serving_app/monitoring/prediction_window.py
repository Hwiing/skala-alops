"""
AIOps 1/3 (#15): 예측·실제값 기록과 최근 윈도우 관리.

왜 리스트 하나(recent_predictions)가 아니라 별도 모듈인가
    - 정답 지연 도착: 실서비스에서는 예측 시점에 정답이 없다. 1주차 정답은 7일 뒤,
      4주차는 28일 뒤에 온다. 그래서 "예측만 먼저 기록 → 정답은 나중에 채우기"가 필요하다.
    - 배치 vs 실시간 구분: batch_test는 과거 데이터라 정답을 같이 넣는다(source="batch").
      실시간 예측은 정답 없이 기록된다(source="live"). 증빙에서 둘을 섞어 말하지 않기 위해 표시한다.
    - 같은 날짜를 다시 보내면(배치 재전송) 중복으로 쌓지 않고 덮어쓴다 → RMSE가 부풀지 않음.
    - FastAPI의 동기(def) 엔드포인트는 스레드풀에서 동시에 돌 수 있어 Lock으로 보호한다.
"""

import threading

from serving_app.monitoring.drift_detector import HORIZONS, WINDOW_SIZE, evaluate

# 정답 확보된 28건 + 아직 정답을 기다리는 최대 28일치(4주차 정답 지연)까지 보관
MAX_RECORDS = WINDOW_SIZE * 2


class PredictionWindow:
    def __init__(self, max_records: int = MAX_RECORDS):
        self._max = max_records
        self._records: list[dict] = []
        self._lock = threading.Lock()

    def record(
        self,
        date: str,
        predicted: list[float],
        naive: float,
        model_version: str | None,
        actual: list[float | None] | None = None,
        source: str = "live",
    ) -> None:
        """예측 1건 기록. 같은 date가 이미 있으면 교체(재전송 중복 방지)."""
        entry = {
            "date": date,
            "predicted": list(predicted),
            "actual": list(actual) if actual is not None else [None] * HORIZONS,
            "naive": naive,
            "model_version": model_version,
            "source": source,
        }
        with self._lock:
            self._records = [r for r in self._records if r["date"] != date]
            self._records.append(entry)
            self._records.sort(key=lambda r: r["date"])
            del self._records[: -self._max]

    def record_batch(self, pairs: list[dict], model_version: str | None) -> None:
        """batch_test의 BatchPair 목록(정답 포함)을 한 번에 기록한다."""
        for p in pairs:
            self.record(p["date"], p["predicted"], p["naive"], model_version, p["actual"], "batch")

    def fill_actual(self, date: str, week: int, value: float) -> bool:
        """지연 도착한 정답 채우기. week는 1~4. 해당 예측이 없으면 False."""
        if not 1 <= week <= HORIZONS:
            raise ValueError(f"week는 1~{HORIZONS}: {week}")
        with self._lock:
            for r in self._records:
                if r["date"] == date:
                    r["actual"][week - 1] = value
                    return True
        return False

    def pairs(self) -> list[dict]:
        with self._lock:
            return [
                dict(r, predicted=list(r["predicted"]), actual=list(r["actual"]))
                for r in self._records
            ]

    def evaluate(self) -> dict:
        return evaluate(self.pairs())

    def clear(self) -> None:
        """새 모델 승격 후 호출 - 이전 모델의 오차로 새 모델을 판정하지 않도록."""
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


# 서버 프로세스 전체가 공유하는 기본 윈도우 (batch_test·retrain_trigger가 같은 객체를 본다)
window = PredictionWindow()
