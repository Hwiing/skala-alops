"""
AIOps 파이프라인 시연용 스텁 서버 (#16 증빙 재현용, 성능 증빙 아님).

실제 LSTM 대신 시나리오 생성식(부분조정 모형)을 아는 스텁 모델을 넣고, fine_tune·reload를 가짜로
바꿔 감지 → 재학습 → 승격 → 알림 → 서빙 교체 흐름만 HTTP로 확인한다. main의 v2 서빙(#34·#46)에서 동작한다.

    STUB_BIAS=40  : 모델이 학습한 관계가 낡은 상황(개념 드리프트)을 흉내 - 예측에 +40원 편향
    AIOPS_ALERT_WEBHOOK_URL=http://127.0.0.1:9019 RETRAIN_COOLDOWN_SECONDS=0 \
        python scripts/aiops_stub_server.py        # 포트 8011
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

from serving_app import model_loader
from serving_app.main import app
from serving_app.monitoring import retrain_trigger as rt

L_PER_BBL, FULL_TAX_WON, PASS_THROUGH = 158.987, 528.75, 0.15


class StubModel:
    version = "champion:3"

    def predict(self, rows):
        def cost(r):
            return r["singapore_diesel_price"] * r["usd_krw"] / L_PER_BBL

        def tax(r):
            return (10 - r["tax_or_supply_feature"]) / 100 * FULL_TAX_WON

        anchor = sum(r["diesel_price"] - cost(r) - tax(r) for r in rows[:60]) / 60
        last = rows[-1]
        target = cost(last) + anchor + tax(last)
        price, path = last["diesel_price"], []
        for _ in range(28):
            price += PASS_THROUGH * (target - price)
            path.append(price)
        bias = float(os.getenv("STUB_BIAS", "0"))  # 낡은 관계(개념 드리프트) 흉내
        return [sum(path[7 * k : 7 * k + 7]) / 7 + bias for k in range(4)]


_next = {"version": 3}


def fake_rows():
    return [{"date": "2024-06-12"}] * 626 + [{"date": "2026-02-28"}]


def fake_fine_tune(rows):
    _next["version"] += 1
    return {
        "promoted": True,
        "status": "promoted",
        "version": str(_next["version"]),
        "rmse": [18.0, 30.0, 41.0, 52.0],
        "naive_rmse": [35.0, 60.0, 80.0, 90.0],
    }


def fake_reload(version):
    model_loader._model_cache.version = f"champion:{version}"
    return {"reloaded": True, "version": f"champion:{version}"}


if __name__ == "__main__":
    model_loader._model_cache = StubModel()
    model_loader.get_model = lambda: model_loader._model_cache
    rt._load_recent_rows, rt._fine_tune = fake_rows, fake_fine_tune
    model_loader.reload_model = fake_reload
    uvicorn.run(app, host="127.0.0.1", port=8011, log_level="warning")
