# 공통 데이터·API·운영 계약

## 적용 범위와 구현 상태

공통 계약 ID는 **`diesel-weekly-v2`**입니다. 다음날 휘발유 v1 계약을 대체합니다.
수치·컬럼·모델 주소는 [`data/contracts.py`](../data/contracts.py), API·AIOps 결과 타입은
[`serving_app/schemas.py`](../serving_app/schemas.py)가 기준입니다. 데이터 생성·업로드·학습은
같은 `validate_daily_rows()`를 사용합니다. 변경 시 이 문서·요청 예시·계약 테스트를 함께 수정합니다.

현재 입력 검증·주간 예측·배치 정답 연결·pyfunc 로더·승격 버전 확인은 구현됐습니다.
드리프트 판정·자동 재학습 트리거·운영자 알림도 `/predict/batch-test`에 연결됐습니다.
실시간 `/predict` 기록과 업로드를 통한 지연 정답 적재도 구현됐습니다. 재학습 데이터는 설정 CSV와 업로드의 유효한 최소 627행 후보 중 마지막 날짜가 가장 최근인 파일에서 선택합니다.
실모델이 없으면 예측은 503입니다.
이 계약 정의가 전체 AIOps 데모 완료를 의미하지는 않습니다.

## 데이터와 모델 경계

| 항목 | 확정 계약 |
|---|---|
| 예측 대상 | 다음 1·2·3·4주 전국 평균 자동차용경유 가격의 주간 평균, 원/L |
| CSV 컬럼 순서 | `date,diesel_price,singapore_diesel_price,usd_krw,tax_or_supply_feature` |
| `date` | `YYYY-MM-DD`, 오래된 날부터 하루 간격. 중복·누락·역순 금지 |
| `diesel_price` | 자동차용경유 원/L, 양수·유한값. 보통휘발유를 이름만 바꿔 사용하지 않음 |
| `singapore_diesel_price` | 싱가포르 경유 0.001% USD/bbl, D-1까지 관측한 최근 값. 두바이유와 다른 상품 |
| `usd_krw` | KRW/USD, D일까지 고시된 최근 값, 양수·유한값 |
| `tax_or_supply_feature` | 경유 기본 탄력세율 375원 대비 인하율 %, 유한값. 공급 차질은 시나리오로 구분 |
| API 입력 | 날짜 포함 최근 **120일**, 정확히 120행. 입력 기준일 D = 마지막 행 날짜 |
| 모델 내부 입력 | 마지막 **28일 × 8피처**. 90일 피처 계산에 필요한 문맥을 포함해 120행을 전달 |
| 모델 출력 | 1주차부터 원/L 숫자 **정확히 4개**, 양수·유한값. 날짜 계산은 서빙 담당 |
| k주 구간 | D+7(k−1)+1 ~ D+7k, 양끝 포함 7일 |
| naive | D일 `diesel_price`를 1~4주 모두에 사용 |
| 배치·업로드 최소 행 | **175행** = 입력 120 + 기준일 짝 28 − 1 + 4주 정답 28 |
| 재학습 최소 행 | 저장된 전체 데이터에서 최근 **627행**. 175행 업로드만으로는 부족 |

CSV의 추가 열은 공통 피처에서 제외합니다. API의 추가 JSON 필드는 모든 중첩 객체에서 거부합니다.
숫자는 NaN·Infinity를 허용하지 않습니다. JSON 날짜에 epoch 숫자·시간 포함 문자열·`YYYYMMDD`는 허용하지 않습니다.
최소 행 수는 작업마다 다릅니다. 직접 `/predict`에 보내는 120행은 업로드 175행 제한과 별개입니다.

실측 선택본은 2008-04-15부터의 확장본입니다. 2012-12-03 이전 국제가격은 경유 0.05%에
첫 60개 겹침 거래일 평균 스프레드 1.479667 USD/bbl를 더한 추정값이며, 접합 여부·관측일은
별도 provenance CSV로 전달합니다. 휴장일은 이전 관측값으로 채우고 미래 값으로 보간하지 않습니다.
출처·관측 시차·원본 검증·재생성은 [`data/README.md`](../data/README.md)를 따릅니다.

정책표는 `data/reference/diesel_fuel_tax_cut.csv`, `diesel_tax_announced.csv`, `price_cap.csv`입니다.
모델은 실행 위치의 정책표를 읽으며, 미래 정책은 예측 기준일까지 발표된 변경만 사용합니다.
정책표 변경은 동일 모델의 예측에도 영향을 주므로 적용한 정책표 버전과 실행 결과를 함께 기록합니다.

## HTTP API

| Method | URL | 요청 / 응답 | 오류 |
|---|---|---|---|
| GET | `/health` | `status`, `contract_version`, `model_loaded`, `model_version`, `model_source`, `loading_mode` | liveness이며 lazy 첫 추론 전 `model_loaded=false`는 정상 |
| POST | `/data/upload` | UTF-8/BOM CSV 최소 175행 → `filename`, `rows`, `filled`(실시간 예측에 채운 주차 정답 수), 채운 정답이 있으면 `drift_check` | 인코딩·컬럼·행 수·데이터 검증 실패 400, 판정은 batch-test와 같은 501/503 |
| GET | `/data/status` | `exists`, 파일명·기간·행 수·가격 범위 | 데이터가 없으면 `exists:false` |
| POST | `/predict` | `PredictRequest` → `PredictResponse`. 성공 시 예측 기록에 정답 없이 남김(`source=live`) | 입력 422, 모델 미준비/모델 출력 계약 위반 503 |
| POST | `/predict/batch-test` | `BatchTestRequest` → `BatchTestResponse` | 입력 422, 모델 미준비/결과 계약 위반 503, AIOps 미구현 501 |
| GET | `/logs` | 로그 파일 목록 | 기존 조회 API |
| GET | `/logs/{filename}` | `name`, `content` | 없는 파일 404, 경로 조작 400 |

`PredictRequest`는 `{sequence: DailyPoint[120]}`입니다. 실행 가능한 합성 요청은
[`examples/predict.json`](../examples/predict.json), 정상 응답 예시는
[`examples/predict-response.json`](../examples/predict-response.json)입니다.

```json
{
  "base_date": "2026-04-30",
  "model_version": "champion:3",
  "predictions": [
    {"horizon_week": 1, "start_date": "2026-05-01", "end_date": "2026-05-07", "predicted_avg_price": 1601.0},
    {"horizon_week": 2, "start_date": "2026-05-08", "end_date": "2026-05-14", "predicted_avg_price": 1602.0},
    {"horizon_week": 3, "start_date": "2026-05-15", "end_date": "2026-05-21", "predicted_avg_price": 1603.0},
    {"horizon_week": 4, "start_date": "2026-05-22", "end_date": "2026-05-28", "predicted_avg_price": 1604.0}
  ]
}
```

`BatchTestRequest`는 `{rows: DailyPoint[175 이상]}`입니다. 실행 가능한 합성 요청은
[`examples/batch-test.json`](../examples/batch-test.json)입니다.
배치는 `rows[i:i+120]`으로 예측하고, 입력 마지막 날 다음날부터 7일씩 평균 내어 정답 4개를 붙입니다.
N행으로 `N−120−28+1`개 짝을 만듭니다. 175행이면 기준일이 연속인 28개 짝이 됩니다.
`predictions`는 모든 짝, AIOps 입력은 최근 28개 짝입니다. 다음 구조의 `BatchPair`를 전달합니다.

```json
{"date": "2026-04-30", "predicted": [1620, 1621, 1622, 1623], "actual": [1623, 1630, 1637, 1644], "naive": 1619}
```

배치 응답은 `{predictions: BatchPair[], drift_check: DriftCheck}`입니다.
`predicted`·`actual`은 각각 정확히 4개이며 같은 주차 순서입니다.
`date`는 정답 도착일이 아니라 **입력 기준일**입니다. 배치 시뮬레이션은 이미 존재하는 정답을
사용합니다. 실운영에서는 1주 정답은 7일 뒤, 전체 4주 정답은 28일 뒤에만 확보할 수 있습니다.
중복 기준일을 실운영 기록으로 재집계하지 않는 정책은 AIOps 담당이 구현합니다.

## MLflow·게이트·재학습

등록 모델은 `DieselPricePredictor`, 서빙 alias는 `champion`입니다.
승격 시 `Production` stage와 `champion` alias를 같은 버전으로 맞춥니다.
서빙은 alias가 가리키는 버전을 확인하고 `models:/DieselPricePredictor/<version>`으로 고정해
`mlflow.pyfunc.load_model()`로 읽습니다. 응답 버전은 `champion:<version>`입니다.
로컬은 `serving_app/models/diesel_pyfunc`를 같은 로더로 읽고 응답 버전은 `local`입니다.
scaler는 모델 아티팩트에 포함하며 별도 v1 `scaler.pkl`을 새 모델에 적용하지 않습니다.

배포 게이트는 다음 조건을 모두 만족해야 합니다.

1. 1주차 RMSE ≤ 50원/L.
2. 1~4주 **각각** RMSE < 같은 기간 naive RMSE.
3. Production이 있으면 같은 검증 구간에서 1~4주 평균 RMSE ≤ Production의 평균 RMSE.

모델·naive·Production RMSE는 각각 정확히 4개, 유한·비음수입니다. `production_rmse=None`만
Production 없는 상태이며 빈 배열은 거부합니다. 실패 시 새 모델을 승격하지 않고 기존 버전을 유지합니다.
50원 기준의 업무 근거와 모델 평가 연구는 [`docs/evidence/06`](evidence/06_데이터분석_결과.md),
[`09`](evidence/09_싱가포르경유_0.05접합_근거.md)를 참고합니다.

`diesel_registry.fine_tune(rows)`는 공통 검증을 통과한 하루 간격 데이터 최소 627행을 받습니다.
Production 가중치로 warm start하고 scaler는 재fit하지 않습니다. 최근 365개 학습 기준일,
학습 타깃과 검증 기준일 사이 28일 간격, 최근 90개 검증 기준일과 마지막 검증의 28일 정답을 확보합니다.
행 부족은 `ValueError("insufficient_data: ...")`입니다.

`FineTuneResult`는 `status`, `promoted`, `rmse[4]`, `naive_rmse[4]`, `production_rmse[4] 또는 null`,
`production_before`, `passed`, `reasons`, `run_id`, `version`을 정의합니다.
선택 메타데이터는 null일 수 있습니다. 승격 시 `version`은 레지스트리 번호 문자열이며 필수입니다.
공통 스키마의 `RegistryVersion`은 MLflow가 반환하는 양의 정수 버전도 입력으로 받습니다.
`production_before`와 `version`은 검증 시 문자열로 정규화하고 JSON 응답에는 항상 문자열로
내보냅니다. 기존 문자열 입력과 null은 유지하며, bool·float·0·음수 정수는 거부합니다.
서빙 버전(`local`, `champion:<version>`)과 `reload.version`은 기존 문자열 계약을 유지합니다.

| 재학습 status | promoted | 의미 |
|---|---|---|
| `promoted` | true | 게이트 통과·Registry 승격. rmse/naive_rmse 4개와 version 필수 |
| `gate_failed` | false | 학습·검증 완료, 게이트 실패. rmse/naive_rmse 4개, 기존 모델 유지 |
| `no_production` | false | warm start할 Production 없음. rmse/naive_rmse는 null |

## AIOps 상태와 서빙 교체

드리프트 계약은 **정답이 확보된 최근 28개 기준일의 1주차 RMSE > max(같은 기간 naive RMSE, 10원/L)**입니다.
28개 미만은 정상으로 간주하지 않고 `insufficient_data`입니다. 탐지 임계치는 배포 게이트 50원과 별개입니다.
`drift_rmse`, `drift_naive_rmse`는 기존 서빙 모델의 탐지 지표입니다.
`rmse[4]`, `naive_rmse[4]`는 재학습 후 독립 검증 지표이므로 화면·로그에서도 구분합니다.

판정은 두 경로에서 실행됩니다. `/predict/batch-test`는 과거 데이터라 정답과 함께 기록(`source=batch`)하고 바로 판정합니다.
실시간 `/predict`는 정답 없이 기록(`source=live`)하고, 이후 `/data/upload`의 일별 가격이 k주 7일을 모두 덮으면
그 주 평균을 정답으로 채운 뒤 판정합니다. 7일 중 하루라도 없으면 그 주는 비워 둡니다. 판정은 최신 모델 버전의 기록만 씁니다.

`DriftCheck.status`는 아래 6개만 사용합니다. 과거 `retrain_triggered`는 더 이상 사용하지 않습니다.
AIOps 담당은 `fine_tune()` 결과를 이 상태로 전달하고, 원인을 `reasons`에 기록합니다.

| status | 의미 |
|---|---|
| `ok` | 충분한 정답으로 판정했으며 드리프트 없음 |
| `insufficient_data` | 판정 짝 부족 또는 재학습 데이터 627행 미만 |
| `promoted` | 재학습 게이트 통과·Registry 승격. 서빙 교체 결과는 별도 `reload` |
| `gate_failed` | 재학습 게이트 실패·기존 모델 유지 |
| `no_production` | Production이 없어 재학습 불가 |
| `retrain_failed` | 재학습 실행 실패. 실패 이유 기록·기존 모델 유지 |

`reload={reloaded, version, error?}`는 승격 뒤 서빙 담당이 추가합니다.
**`promoted=true`와 실제 새 모델 서빙은 다른 단계**입니다. `MODEL_SOURCE=mlflow`에서 승격 번호와
실제 로드한 번호가 같아야 `reloaded=true`, 응답 버전은 `champion:<승격 번호>`가 됩니다.
이때만 이전 모델의 예측 기록을 비웁니다. local 모드·버전 누락·불일치·로드 실패는 교체 실패이며
현재 모델과 기록을 유지하고 `error`를 남깁니다. 클라이언트는 모르는 상태를 정상으로 표시하지 않습니다.

감지 → `[WARN]` → 재학습 시작 `[INFO]` → 승격 여부 → 실제 교체 결과를 `logs/aiops.log`에 기록합니다.
운영자 알림 채널·수신자, 중복 트리거 억제, 재학습 실패 복구는 AIOps 담당의 구현·검증 항목입니다.
최종 통합 증빙은 가짜 reload 화면과 구분하여 실제 `/predict` 새 버전 응답으로 확인합니다.

## 담당별 인계와 검증

| 담당 | 공통 입력/출력과 완료 증빙 |
|---|---|
| 데이터 | 경유 5컬럼·원본/가공 SHA-256·provenance·120일 입력+28일 정답 정렬·팀 전달 위치 |
| 모델 | pyfunc 입력 120행/출력 4개·scaler 포함·주차별 RMSE/naive/Production·627행 재학습 결과 |
| 서빙 | 422 입력 검증·175행 배치·503 미준비·승격 버전 확인·실패 시 캐시/기록 유지 |
| AIOps | 28개 기준일 판정·6개 상태·627행 재학습 연결·알림/로그·중복 억제 |
| UI·PM | 주차별 결과·탐지/재학습 RMSE 구분·모든 실패 상태·실제 버전 전환·Docker 실행 증빙 |

`make lint test`는 공통 상수·CSV/API 날짜/수치·4주 출력·상태 모순·실패 시 기록 유지와 예시 파일을
검증합니다. TensorFlow/MLflow 실모델 학습·Docker·운영자 알림 증빙은 별도 통합 실행으로 확보합니다.

## Docker 실행·영속화

단일 컨테이너(`serving_app/docker-compose.yml`, 포트 8000). 런타임 데이터는 이름 있는 볼륨에 둡니다.

| 볼륨 | 컨테이너 경로 | 내용 |
|---|---|---|
| `uploads` | `/app/data/uploads` | 대시보드·`/data/upload`로 올린 CSV |
| `processed` | `/app/data/processed` | 학습용 경유 CSV `diesel_features_2008_spliced.csv` (호스트에서 `cp`로 넣음) |
| `models` | `/app/serving_app/models` | `diesel/`(seed별 LSTM·scaler·meta), `diesel_pyfunc/`(`MODEL_SOURCE=local`용 pyfunc). scaler는 모델 폴더 안에 포함 |
| `logs` | `/app/logs` | `aiops.log` |
| `mlflow-db` | `/app/runtime` | `mlflow.db` (`MLFLOW_TRACKING_URI=sqlite:////app/runtime/mlflow.db`) |
| `mlruns` | `/app/mlruns` | MLflow 아티팩트(등록된 pyfunc) |

- `restart`·`down`·재빌드 후에도 유지되고, `down -v`로만 삭제됩니다. `restart: unless-stopped`.
- 이미지에는 데이터·모델을 넣지 않습니다(`data/processed`는 `.dockerignore` 제외). v2 모델은 실제 경유 CSV가 필요하며 `data/sample_diesel_prices.csv`(120행)로는 학습할 수 없습니다.
- 호스트의 `mlflow.db`·`mlruns`는 공유하지 않습니다. 아티팩트 경로가 호스트 절대경로로 기록되어 컨테이너에서 찾을 수 없으므로 컨테이너 안에서 학습·등록합니다.
- AIOps 설정은 호스트 환경변수로 넘깁니다(예: `AIOPS_ALERT_WEBHOOK_URL=<url> MODEL_SOURCE=mlflow dc up -d`).
  `AIOPS_ALERT_WEBHOOK_URL`(비우면 `aiops.log`의 `[ALERT]`만), `AIOPS_ALERT_TIMEOUT`(5초), `AIOPS_ALERT_DEDUP_SECONDS`(3600),
  `RETRAIN_COOLDOWN_SECONDS`(600), `DIESEL_DATA_CSV`(재학습 데이터, 기본 `data/processed/diesel_features_2008_spliced.csv`).
- 재학습 승격 후 자동 교체는 `MODEL_SOURCE=mlflow`에서만 됩니다. `local`은 레지스트리 버전을 서빙하지 않아 `reload`가 거절됩니다.
- 재학습은 `DIESEL_DATA_CSV`와 `data/uploads/*.csv`를 공통 계약으로 검증하고, 627행 이상인 후보 중 마지막 날짜가 가장 최근인 파일의 마지막 627행을 읽습니다. 날짜가 같으면 설정 CSV가 우선이며 업로드 시각은 선택 기준이 아닙니다. 행 부족·검증 실패 파일과 저장 중인 `.pending` 파일은 제외합니다. 후보가 없으면 `insufficient_data`입니다.
- 모델 버전·판정 기간·선택된 627행의 SHA-256이 같으면 확정 결과(`promoted`, `gate_failed`)를 재사용합니다. 데이터 변경은 쿨다운 후 새로 학습합니다. 데이터 부족·학습 예외·Production 없음은 영구 캐시하지 않고 쿨다운 후 같은 판정도 재시도합니다. 동시 실행 방지는 결과 기록까지 하나의 잠금으로 보호합니다.
- 업로드의 검증·저장·지연 정답 연결·재학습은 작업 스레드에서 실행합니다. 업로드 응답은 판정 완료까지 기다리지만 그동안 이벤트 루프는 다른 요청을 처리합니다. CSV는 저장 완료 후 원자적으로 공개합니다.
- 운영자 기본 수신함은 `AIOPS_ALERT_FILE=logs/alerts.jsonl`이며 기존 logs 볼륨에 보관합니다. `GET /logs/alerts.jsonl`로 조회하고, 승격(`promoted`)과 실제 교체(`reloaded`, `reload_failed`)를 별도 알림으로 기록합니다. 이 알림 상태는 `DriftCheck.status` 6개 상태와 별개입니다. `AIOPS_ALERT_WEBHOOK_URL`을 지정하면 외부 채널에도 전송하며 실패 시 수신함 기록과 HTTP 결과는 유지합니다.
- `DIESEL_DATA_SYNTHETIC=true`는 합성 검증 실행에서만 사용합니다. fine-tuning의 로그와 MLflow params에 합성 여부를 기록하며 기본은 `false`입니다. 실측·합성 데이터를 한 저장소에서 혼용하지 않습니다.
- 모델이 없으면 서버는 뜨고 `/predict`만 503입니다(기본 lazy). eager는 시작 시 로드 실패가 바로 드러나지만, 모델이 준비된 뒤 전환합니다.

```bash
dc() { docker compose -f serving_app/docker-compose.yml "$@"; }  # bash·zsh 공통
dc up -d --build                                                         # 빌드 + 실행 (lazy/local, /predict 503)
# 1) 학습 데이터 전달 (호스트에서 data/README.md 절차로 만든 실제 CSV)
dc cp data/processed/diesel_features_2008_spliced.csv serving-app:/app/data/processed/
# 2-a) local 모델: serving_app/models/diesel/, diesel_pyfunc/ 생성 → 재시작 없이 다음 /predict부터 사용(lazy)
dc exec serving-app python scripts/train_diesel_baseline.py
# 2-b) MLflow: 기록 → 배포 게이트 → 통과 시 DieselPricePredictor 등록·Production·alias champion
dc exec serving-app python serving_app/diesel_registry.py
# 3) champion 서빙으로 전환
MODEL_SOURCE=mlflow dc up -d
curl -s localhost:8000/health                                            # model_version: champion:<버전>
```
