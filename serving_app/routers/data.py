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

router = APIRouter(prefix="/data")

REQUIRED_COLUMNS = set(CSV_COLUMNS)
MIN_ROWS = (
    BATCH_MIN_ROWS  # 배치 시뮬레이션 1회(입력 120 + 짝 28 − 1 + 4주 정답 28)에 필요한 최소 행 수
)


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
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

    return {"filename": os.path.basename(dest), "rows": len(rows)}


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
