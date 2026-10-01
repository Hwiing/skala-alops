"""경유 v2 CSV 업로드(최소 175행)·최신 데이터 상태 조회. 공통 검증은 data/contracts.py."""

import csv
import io
import os
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.contracts import BATCH_MIN_ROWS, CSV_COLUMNS
from data.diesel import validate_diesel_rows
from data.diesel_features import load_diesel_rows
from data.storage import UPLOAD_DIR, latest_upload
from serving_app.routers import predict as predict_router

router = APIRouter(prefix="/data")

REQUIRED_COLUMNS = set(CSV_COLUMNS)
MIN_ROWS = (
    BATCH_MIN_ROWS  # 배치 시뮬레이션 1회(입력 120 + 짝 28 − 1 + 4주 정답 28)에 필요한 최소 행 수
)


# def(동기)라서 FastAPI가 스레드풀에서 실행한다. 업로드 뒤 판정이 재학습(TensorFlow, 수십 초)까지
# 이어져도 이벤트 루프를 막지 않아 /health·/predict는 계속 응답한다.
@router.post("/upload")
def upload(file: UploadFile = File(...)):
    raw = file.file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.")

    reader = csv.DictReader(io.StringIO(text))
    if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
        raise HTTPException(400, f"CSV에 {sorted(REQUIRED_COLUMNS)} 컬럼이 모두 있어야 합니다.")
    rows = list(reader)
    if len(rows) < MIN_ROWS:
        raise HTTPException(400, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다.")

    try:
        validate_diesel_rows(rows)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(400, f"CSV 데이터 검증 실패: {exc}") from exc

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"diesel_{uuid4().hex}.csv")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    # 업로드는 실시간 예측의 지연 정답이기도 하다. 새로 채운 정답이 있거나, 이전 판정·교체가
    # 끝나지 않았으면(판정 오류·교체 실패) 판정한다. 같은 판정은 AIOps가 이전 결과를 재사용하므로
    # 재학습 없이 교체만 다시 시도된다.
    filled = predict_router.fill_live_actuals(rows)
    result = {"filename": os.path.basename(dest), "rows": len(rows), "filled": filled}
    if filled or predict_router.judgement_pending():
        try:
            result["drift_check"] = predict_router.judge_and_swap()
        except HTTPException as exc:  # 파일 저장·정답 적재는 끝났으므로 업로드는 성공으로 둔다
            result["drift_check_error"] = {"status_code": exc.status_code, "detail": exc.detail}
    return result


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}

    rows = load_diesel_rows(path)
    closes = [r["diesel_price"] for r in rows]
    return {
        "exists": True,
        "filename": os.path.basename(path),
        "rows": len(rows),
        "start_date": rows[0]["date"],
        "end_date": rows[-1]["date"],
        "min_price": min(closes),
        "max_price": max(closes),
    }
