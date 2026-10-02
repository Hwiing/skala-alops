"""
#10 경유 모델 MLflow 기록 → 배포 게이트 → 등록·Production 승격 (contracts.md v2).

- 모델은 pyfunc 하나(DieselPricePredictor): seed별 LSTM + 학습 구간 scaler + 정책 규칙.
  입력: DataFrame 1건 = 최근 120행(date + 경유 4컬럼, 오래된 날 → 최근 날)
  출력: 1~4주 평균 경유가 float 4개(원/L, 1주차부터). k주 날짜 = base_date+7(k−1)+1 ~ base_date+7k (서빙이 계산)
  주소: 승격 시 stage "Production"과 alias "champion"을 같이 붙인다
        (models:/DieselPricePredictor/Production 또는 models:/DieselPricePredictor@champion, 같은 버전).
  로컬: save_local_pyfunc()로 저장한 폴더를 mlflow.pyfunc.load_model(<폴더>)로 같은 방식으로 읽는다.
  정책표는 실행 위치의 data/reference/를 읽는다.
- 게이트(serving_app/diesel_gate.py): 1주차 RMSE ≤ 50 AND 1~4주 모두 < naive AND 1~4주 평균 오차(RMSE) ≤ Production.
  Production은 새 모델과 같은 검증 구간으로 다시 평가해 비교한다.
- 미통과 시 등록하지 않는다. Production이 없으면 "서비스할 모델 없음", 있으면 "기존 버전 유지"로 구분해 기록.
- fine_tune(rows) (#11): Production에서 warm start, 최근 365일 학습, 학습과 겹치지 않는 최근 90일로 같은 게이트.
  반환 {promoted, rmse, naive_rmse, version?, status, reasons, ...}
    status: "promoted" | "gate_failed" | "no_production"
    예외: 행이 FINETUNE_MIN_ROWS(627)보다 적으면 ValueError("insufficient_data: ...")

실행: python serving_app/diesel_registry.py [--csv data/processed/diesel_features_2008_spliced.csv]
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from data.contracts import MODEL_ALIAS as ALIAS
from data.contracts import MODEL_NAME
from data.diesel_features import (  # noqa: E402
    DIESEL_COLUMNS,
    FEATURES,
    FINETUNE_MIN_ROWS,
    INPUT_DAYS,
    SEQ_LEN,
    DailyFrame,
    load_diesel_rows,
    split_finetune,
)
from data.diesel_policy import PASS_DAY0, PASS_DAYS  # noqa: E402
from serving_app.diesel_gate import WEEK1_RMSE_MAX, check_gate  # noqa: E402
from serving_app.diesel_model import DieselForecaster  # noqa: E402
from serving_app.diesel_training import (  # noqa: E402
    evaluate,
    finetune,
    fit,
    report,
    split_holdout,
)
from serving_app.schemas import FineTuneResult

# warm start 강도. base 학습(3e-4·40 epoch)보다 약하게 둬서 17년치로 배운 것을 유지한다.
# 더 세게(1e-3·30 epoch 등) 하면 5개 시점 중 4곳에서 Production보다 나빠졌다 (evidence/11).
# 실험·시연용으로만 환경변수로 바꾼다.
FINE_TUNE_EPOCHS = int(os.getenv("FINE_TUNE_EPOCHS", "10"))
FINE_TUNE_LR = float(os.getenv("FINE_TUNE_LR", "1e-4"))
DEFAULT_CSV = "data/processed/diesel_features_2008_spliced.csv"


class DieselPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow 래퍼. 입력 DataFrame(date + 경유 4컬럼, 최근 120행) → 1~4주 평균가 float 4개."""

    def load_context(self, context):
        self.forecaster = DieselForecaster.load(context.artifacts["model_dir"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> list[float]:
        frame = model_input.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        rows = frame[["date", *DIESEL_COLUMNS]].to_dict("records")
        return self.forecaster.predict(rows)


def save_local_pyfunc(forecaster: DieselForecaster, path: str, meta: dict | None = None):
    """MODEL_SOURCE=local용. MLflow 레지스트리 없이 같은 pyfunc 형식으로 폴더에 저장한다."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)
    with tempfile.TemporaryDirectory() as tmp:
        forecaster.save(tmp, meta)
        mlflow.pyfunc.save_model(path, python_model=DieselPyfunc(), artifacts={"model_dir": tmp})


def production_version(client: MlflowClient) -> str | None:
    try:
        versions = client.get_latest_versions(MODEL_NAME, ["Production"])
    except MlflowException:  # 등록 모델 이름이 아직 없음
        return None
    return str(versions[0].version) if versions else None  # sqlite 스토어는 int를 돌려준다


def load_forecaster(version: str) -> DieselForecaster:
    return (
        mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}").unwrap_python_model().forecaster
    )


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _log_training(forecaster, meta: dict) -> str:
    """현재 MLflow run에 학습 지표·모델을 기록한다. 배포 판단과 등록은 수행하지 않는다."""
    mlflow.log_params(
        {
            "mode": meta.get("mode", "scratch"),
            "data": meta["data"],
            "data_sha256": meta["data_sha256"],
            "data_period": " ~ ".join(meta["data_period"]),
            "rows": meta["rows"],
            "synthetic": meta.get("synthetic", False),
            "features": ",".join(FEATURES),
            "seq_len": SEQ_LEN,
            "input_days": INPUT_DAYS,
            "seeds": ",".join(map(str, meta["seeds"])),
            "epochs": ",".join(map(str, meta["epochs"])),
            "learning_rate": meta["learning_rate"],
            "base_version": meta.get("base_version", "none"),
            "train_target_period": " ~ ".join(meta["train_target_period"]),
            "validation_period": " ~ ".join(meta["validation_period"]),
            "policy_pass_day0": PASS_DAY0,
            "policy_pass_days": PASS_DAYS,
        }
    )
    for k, (m, n) in enumerate(zip(meta["rmse"], meta["naive_rmse"]), start=1):
        mlflow.log_metrics({f"rmse_w{k}": m, f"naive_rmse_w{k}": n})
    with tempfile.TemporaryDirectory() as tmp:
        forecaster.save(tmp, meta)
        model = mlflow.pyfunc.log_model(
            name="model", python_model=DieselPyfunc(), artifacts={"model_dir": tmp}
        )
    return model.model_uri


def log_and_gate(forecaster, frame, val_idx, meta: dict, run_name: str) -> dict:
    """같은 검증 구간으로 Production을 재평가해 게이트 판단 → 기록 → 통과 시 등록·승격. #11 fine-tuning도 사용."""
    client = MlflowClient()
    before = production_version(client)
    production_rmse = evaluate(load_forecaster(before), frame, val_idx)["rmse"] if before else None
    gate = check_gate(meta["rmse"], meta["naive_rmse"], production_rmse)
    result = {
        "rmse": meta["rmse"],
        "naive_rmse": meta["naive_rmse"],
        "production_rmse": production_rmse,
        "production_before": before,
        "passed": gate["passed"],
        "reasons": gate["reasons"],
        "promoted": False,
    }
    with mlflow.start_run(run_name=run_name) as run:
        result["run_id"] = run.info.run_id
        mlflow.log_params({"gate_week1_max": WEEK1_RMSE_MAX, "production_before": before or "none"})
        for k, v in enumerate(production_rmse or [], start=1):
            mlflow.log_metric(f"production_rmse_w{k}", v)
        mlflow.set_tags(
            {
                "gate": "passed" if gate["passed"] else "failed",
                "gate_reasons": "; ".join(gate["reasons"]),
            }
        )
        _log_training(forecaster, meta)
        if gate["passed"]:
            v = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
            client.transition_model_version_stage(
                MODEL_NAME, v.version, "Production", archive_existing_versions=True
            )
            client.set_registered_model_alias(MODEL_NAME, ALIAS, v.version)
            result.update(promoted=True, version=str(v.version))
            print(
                f"[GATE PASSED] {MODEL_NAME} v{v.version} → Production (이전: {before or '없음'})"
            )
        elif before is None:
            print(
                f"[GATE FAILED] {'; '.join(gate['reasons'])} → 등록 안 함. Production 없음: 서비스할 모델 없음"
            )
        else:
            print(
                f"[GATE FAILED] {'; '.join(gate['reasons'])} → 등록 안 함. 기존 Production v{before} 유지"
            )
    return result


def _fit_initial(csv_path: str, holdout_days: int, seeds, synthetic: bool):
    rows = load_diesel_rows(csv_path)
    frame = DailyFrame(rows)
    train_idx, val_idx = split_holdout(frame, holdout_days)
    forecaster, info = fit(frame, train_idx, list(seeds))
    meta = {
        "mode": "scratch",
        "data": csv_path,
        "data_sha256": file_sha256(csv_path),
        "synthetic": synthetic,
        "data_period": [frame.dates[0].isoformat(), frame.dates[-1].isoformat()],
        "rows": len(rows),
        **info,
        **evaluate(forecaster, frame, val_idx),
    }
    print(report(meta))
    return forecaster, frame, val_idx, meta


def train_initial(
    csv_path: str = DEFAULT_CSV, holdout_days: int = 365, seeds=(42, 7, 2026), synthetic=False
):
    """학습·독립 검증·MLflow 저장 전용 함수. 배포는 train_and_register()를 사용한다."""
    forecaster, _, _, meta = _fit_initial(csv_path, holdout_days, seeds, synthetic)
    with mlflow.start_run(run_name="diesel-initial-train") as run:
        mlflow.set_tags({"workflow": "initial-training", "training_only": "true"})
        model_uri = _log_training(forecaster, meta)
        return {
            "rmse": meta["rmse"],
            "naive_rmse": meta["naive_rmse"],
            "run_id": run.info.run_id,
            "model_uri": model_uri,
        }


def train_and_register(
    csv_path: str = DEFAULT_CSV, holdout_days: int = 365, seeds=(42, 7, 2026), synthetic=False
):
    """대시보드와 운영자 CLI의 초기 학습·게이트·등록 경로."""
    forecaster, frame, val_idx, meta = _fit_initial(csv_path, holdout_days, seeds, synthetic)
    return log_and_gate(forecaster, frame, val_idx, meta, run_name="diesel-base-train")


def fine_tune(rows: list[dict], *, synthetic: bool = False) -> dict:
    """#11 드리프트 재학습. rows: 최근 데이터(하루 간격, 최소 FINETUNE_MIN_ROWS행)."""
    from data.contracts import validate_daily_rows

    frame = DailyFrame(validate_daily_rows(rows))
    train_idx, val_idx = split_finetune(frame)  # 부족하면 ValueError("insufficient_data: ...")
    version = production_version(MlflowClient())
    if version is None:
        print(
            "[FINE-TUNE SKIPPED] Production 없음 → warm start 불가. 먼저 train_and_register()로 base 모델을 배포하세요"
        )
        return FineTuneResult(
            promoted=False, status="no_production", reasons=["Production 없음"]
        ).model_dump(exclude_none=False)
    forecaster = load_forecaster(version)
    info = finetune(forecaster, frame, train_idx, FINE_TUNE_EPOCHS, FINE_TUNE_LR)
    meta = {
        "mode": "fine-tune",
        "synthetic": synthetic,
        "data": f"rows:{len(rows)}",
        "data_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "data_period": [frame.dates[0].isoformat(), frame.dates[-1].isoformat()],
        "rows": len(rows),
        "base_version": version,
        **info,
        **evaluate(forecaster, frame, val_idx),
    }
    print(report(meta))
    result = log_and_gate(forecaster, frame, val_idx, meta, run_name="diesel-fine-tune")
    result["status"] = "promoted" if result["promoted"] else "gate_failed"
    return FineTuneResult.model_validate(result).model_dump(exclude_none=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--holdout-days", type=int, default=365)
    ap.add_argument("--seeds", default="42,7,2026")
    ap.add_argument(
        "--fine-tune", action="store_true", help="Production에서 이어서 최근 데이터로 재학습 (#11)"
    )
    ap.add_argument(
        "--synthetic", action="store_true", help="합성 데이터면 표시 (성능 증빙에 쓰지 않음)"
    )
    args = ap.parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    if args.fine_tune:
        print(fine_tune(load_diesel_rows(args.csv)[-FINETUNE_MIN_ROWS:], synthetic=args.synthetic))
        return
    train_and_register(
        args.csv, args.holdout_days, [int(s) for s in args.seeds.split(",")], args.synthetic
    )


if __name__ == "__main__":
    main()
