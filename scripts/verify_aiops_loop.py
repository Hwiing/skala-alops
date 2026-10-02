"""격리된 합성 fixture로 실제 TF 재학습·MLflow 승격·HTTP 교체·알림 수신을 검증한다.

실행: python scripts/verify_aiops_loop.py [--out /tmp/aiops-result.json]
처음의 stale Production은 의도적으로 준비한 fixture다. 새 버전은 실제 fine_tune과
기존 배포 게이트를 통과해야 한다. 실측 성능 증빙으로 사용하지 않는다.
"""

import argparse
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")


def fixture_rows(start: date) -> list[dict]:
    return [
        {
            "date": (start + timedelta(days=i)).isoformat(),
            "diesel_price": 1500.0 + 3 * i,
            "singapore_diesel_price": 90.0,
            "usd_krw": 1400.0,
            "tax_or_supply_feature": 0.0,
        }
        for i in range(700)
    ]


def csv_text(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def request(base, path, body=None, content_type="application/json"):
    req = urllib.request.Request(
        base + path,
        data=body,
        headers={"Content-Type": content_type},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def verify(runtime: Path) -> dict:
    import mlflow
    import numpy as np
    import uvicorn
    from mlflow.tracking import MlflowClient
    from tensorflow import keras

    from data.contracts import MODEL_ALIAS, MODEL_NAME
    from data.diesel_features import DailyFrame
    from serving_app.diesel_model import DieselForecaster, build_model
    from serving_app.diesel_registry import DieselPyfunc
    from serving_app.diesel_training import to_arrays

    shutil.copytree(ROOT / "data/reference", runtime / "data/reference")
    rows = fixture_rows(date(2014, 1, 1))
    new_rows = fixture_rows(date(2014, 1, 2))
    source = runtime / "data/processed/base.csv"
    source.parent.mkdir(parents=True)
    source.write_text(csv_text(rows))
    os.environ.update(
        MODEL_SOURCE="mlflow",
        LOADING_MODE="lazy",
        DIESEL_DATA_CSV=str(source),
        DIESEL_DATA_SYNTHETIC="true",
        MLFLOW_TRACKING_URI=f"sqlite:///{runtime / 'mlflow.db'}",
        AIOPS_ALERT_FILE="logs/alerts.jsonl",
    )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    experiment = client.create_experiment(
        "synthetic-aiops-loop", artifact_location=str(runtime / "artifacts")
    )
    mlflow.set_experiment(experiment_id=experiment)
    keras.utils.set_random_seed(42)
    frame = DailyFrame(rows)
    scaler, _, _, _ = to_arrays(frame, list(range(119, 600)))
    model = build_model()
    # Dense(16) 상수 활성화와 약한 음수 출력을 가진 낡은 모델 fixture.
    dense, output = model.layers[-2:]
    weights = dense.get_weights()
    dense.set_weights([np.zeros_like(weights[0]), np.ones_like(weights[1])])
    weights = output.get_weights()
    output.set_weights([np.zeros_like(weights[0]), np.full_like(weights[1], -0.02)])
    stale = DieselForecaster([model], scaler)
    model_dir = runtime / "seed-model"
    stale.save(str(model_dir))
    with mlflow.start_run(run_name="synthetic-stale-production-fixture") as run:
        mlflow.set_tag("synthetic", "true")
        mlflow.pyfunc.log_model(
            name="model", python_model=DieselPyfunc(), artifacts={"model_dir": str(model_dir)}
        )
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
    client.transition_model_version_stage(
        MODEL_NAME, version.version, "Production", archive_existing_versions=True
    )
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version.version)

    received = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    webhook = HTTPServer(("127.0.0.1", 0), Receiver)
    webhook_thread = threading.Thread(target=webhook.serve_forever, daemon=True)
    webhook_thread.start()
    os.environ["AIOPS_ALERT_WEBHOOK_URL"] = f"http://127.0.0.1:{webhook.server_port}"
    from serving_app.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            if time.monotonic() > deadline or not server_thread.is_alive():
                raise RuntimeError("HTTP server startup failed")
            time.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        predict_body = json.dumps({"sequence": new_rows[-120:]}).encode()
        before = request(base, "/predict", predict_body)
        boundary = "aiops-verification-boundary"
        upload_body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="new.csv"'
            f"\r\nContent-Type: text/csv\r\n\r\n{csv_text(new_rows)}\r\n--{boundary}--\r\n"
        ).encode()
        upload = request(
            base, "/data/upload", upload_body, f"multipart/form-data; boundary={boundary}"
        )
        batch_body = json.dumps({"rows": new_rows[-175:]}).encode()
        batch = request(base, "/predict/batch-test", batch_body)
        drift = batch["drift_check"]
        assert drift["status"] == "promoted", drift
        assert drift["reload"]["reloaded"], drift
        after = request(base, "/predict", predict_body)
        health = request(base, "/health")
        assert before["model_version"] == "champion:1"
        assert after["model_version"] == health["model_version"] == "champion:2"
        repeated = request(base, "/predict/batch-test", batch_body)["drift_check"]
        assert repeated["status"] == "ok", repeated
        inbox = request(base, "/logs/alerts.jsonl")
        inbox_events = [json.loads(line) for line in inbox["content"].splitlines()]
        assert [e["status"] for e in inbox_events] == ["promoted", "reloaded"]
        assert [p["alert"]["status"] for p in received] == ["promoted", "reloaded"]
        return {
            "synthetic": True,
            "seeded_production_fixture": True,
            "mlflow": mlflow.__version__,
            "upload": upload,
            "before": before,
            "drift_check": drift,
            "after": after,
            "health": health,
            "repeat": repeated,
            "operator_inbox": inbox_events,
            "webhook_received": len(received),
            "aiops_log": (runtime / "logs/aiops.log").read_text(),
        }
    finally:
        server.should_exit = True
        server_thread.join(10)
        webhook.shutdown()
        webhook_thread.join(5)
        webhook.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out.resolve() if args.out else None
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="skala-aiops-") as tmp:
        runtime = Path(tmp)
        try:
            os.chdir(runtime)
            result = verify(runtime)
        finally:
            os.chdir(cwd)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
