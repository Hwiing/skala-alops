"""
대시보드 "운영 현황" 탭용 - MLflow 레지스트리 버전과 재학습 run 이력을 읽기 전용으로 노출한다 (#56).

게이트를 통과한 run만 레지스트리에 등록되므로(diesel_registry.log_and_gate), 등록 버전만 보면
게이트 실패 시도가 보이지 않는다. 그래서 gate 태그가 붙은 run을 기준으로 나열하고,
등록된 run에는 버전·단계·alias를 붙인다. 학습·승격 로직은 건드리지 않는다.
"""

import os
from datetime import datetime, timezone

from fastapi import APIRouter

from data.contracts import MODEL_ALIAS, MODEL_NAME
from serving_app import model_loader

router = APIRouter()

MAX_RUNS = 50  # ponytail: 최근 50개만. 이력이 더 필요하면 페이지네이션 추가


def _iso(ms) -> str | None:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat() if ms else None


def _weekly(metrics: dict, prefix: str) -> list[float]:
    return [metrics[f"{prefix}{k}"] for k in range(1, 5) if f"{prefix}{k}" in metrics]


def _entry(run, v, aliases: dict) -> dict:
    params, metrics, tags = run.data.params, run.data.metrics, run.data.tags
    before = params.get("production_before")
    return {
        "run_id": run.info.run_id,
        "started_at": _iso(run.info.start_time),
        "mode": params.get("mode"),
        "base_version": None if before in (None, "none") else before,
        "data_period": params.get("data_period"),
        "rmse": _weekly(metrics, "rmse_w"),
        "naive_rmse": _weekly(metrics, "naive_rmse_w"),
        "gate": tags.get("gate"),
        "gate_reasons": tags.get("gate_reasons") or None,
        "version": str(v.version) if v else None,
        "stage": v.current_stage if v else None,
        "aliases": [a for a, n in aliases.items() if v and str(n) == str(v.version)],
        "registered_at": _iso(v.creation_timestamp) if v else None,
    }


def _registry() -> tuple[list[dict], dict | None]:
    """(최근 run 이력, champion 버전 항목). champion은 이력 길이와 무관하게 alias에서 직접 찾는다."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in client.search_experiments()],
        filter_string="tags.gate LIKE '%'",
        order_by=["attributes.start_time DESC"],
        max_results=MAX_RUNS,
    )
    versions = {v.run_id: v for v in client.search_model_versions(f"name='{MODEL_NAME}'")}
    # 버전 검색 결과는 aliases를 채우지 않으므로 등록 모델에서 alias → 버전을 따로 읽는다
    aliases = client.get_registered_model(MODEL_NAME).aliases if versions else {}
    history = [_entry(run, versions.get(run.info.run_id), aliases) for run in runs]
    champion = str(aliases[MODEL_ALIAS]) if MODEL_ALIAS in aliases else None
    production = next((h for h in history if h["version"] == champion), None)
    if champion and production is None:  # 최근 MAX_RUNS 밖으로 밀려난 champion run
        v = next((v for v in versions.values() if str(v.version) == champion), None)
        production = _entry(client.get_run(v.run_id), v, aliases) if v else None
    return history, production


@router.get("/models")
def list_models():
    model = model_loader._model_cache
    body = {
        "source": os.getenv("MODEL_SOURCE", "local"),
        "serving_version": model.version if model is not None else None,
        "production_version": None,
        "production": None,
        "history": [],
    }
    if body["source"] != "mlflow":
        return body  # local 모드: 레지스트리 없음
    try:
        body["history"], body["production"] = _registry()
    except Exception as exc:  # 대시보드 조회 실패가 500으로 화면을 깨지 않게 원인만 돌려준다
        body["error"] = f"{type(exc).__name__}: {exc}"
        return body
    body["production_version"] = body["production"]["version"] if body["production"] else None
    return body
