# 공통 데이터·API·운영 계약

## v2 경유 주간 예측 (#28 합의 대상)

v2가 합의·구현되기 전까지 실제 코드는 아래 **v1(휘발유 다음날)** 을 따릅니다. 각 담당은 자기 v2 PR이 main에 들어갈 때 v1 항목을 지웁니다.
근거: [`docs/evidence/06`](evidence/06_데이터분석_결과.md) §11~13, [`09`](evidence/09_싱가포르경유_0.05접합_근거.md). LSTM이 정상기(2016·19·22·25)와 2026 모두 1~4주차에서 naive보다 RMSE가 낮고, 정상기 1~4주·2026 2~4주는 Diebold-Mariano 검정으로 유의(p<0.05)합니다.

| 항목 | v1 (현재) | v2 (변경) | 담당 |
|---|---|---|---|
| 예측 대상 | 다음날 휘발유 가격 | **다음 1·2·3·4주 경유 평균가**(원/L) 4개를 한 번에 출력. k주 = D+7(k−1)+1 ~ D+7k일, D = 마지막 입력일 | 전원 |
| CSV 컬럼 | `date,gasoline_price,crude_oil_price,usd_krw,tax_or_supply_feature` | `date,diesel_price,singapore_diesel_price,usd_krw,tax_or_supply_feature` | 유나 |
| 국제가격 | 두바이유, D-1 | 싱가포르 경유 0.001%(USD/bbl), D-1. 2012-12-02 이전은 0.05% + 1.479667 USD/bbl 접합 | 유나 |
| 기간 | 2023-09 ~ | **2008-04-15 ~** | 유나 |
| 정책표 | 없음 | `data/reference/diesel_fuel_tax_cut.csv`(유류세), `data/reference/price_cap.csv`(최고가격: 시행일·종료일·상한·발표일). 모델이 예측 때 읽음. 정책이 발표되면 표만 갱신, 재학습 불필요 | 유나(세금), 소영(상한) |
| API 입력 | `sequence` 20개, 날짜 없음 | **최근 120일**, 행마다 `date` 포함(오래된 날 → 최근 날). 정책표 조회에 날짜 필요 | 준형 |
| API 출력 | `predicted_price` | `predictions`: `[{horizon_week, start_date, end_date, predicted_avg_price}] × 4`, `base_date`, `model_version`. 모델은 숫자 4개만 내고, 날짜는 서빙이 계산: k주 = `base_date`+7(k−1)+1 ~ `base_date`+7k, `base_date` = 마지막 입력일 | 준형 |
| 모델 | `GasolinePricePredictor`(Keras) | `DieselPricePredictor`, MLflow pyfunc 1개(LSTM + scaler + 정책 규칙). **`predict(df)`**: 입력 DataFrame 1건 = 최근 120행(`date` + 피처 4개, 오래된 날 → 최근 날), 출력 = 1~4주 평균가 float 4개(원/L, 1주차부터) | 소영 → 준형 |
| 모델 주소 | `models:/GasolinePricePredictor/Production` | 승격 시 stage `Production`과 alias `champion`을 같은 버전에 같이 붙임 → `models:/DieselPricePredictor/Production` = `models:/DieselPricePredictor@champion`. `MODEL_SOURCE=local`은 `mlflow.pyfunc.load_model("serving_app/models/diesel_pyfunc")`로 같은 방식으로 읽음 | 소영·준형 |
| naive | 직전 날 가격 | 마지막 입력일 가격을 1~4주 모두에 사용 | 소영 |
| 배포 게이트 | `RMSE ≤ 10 AND RMSE < naive` | **1주차 `RMSE ≤ 50` AND 1~4주 모두 `RMSE < naive_rmse` AND 1~4주 평균 `RMSE ≤ Production 1~4주 평균 RMSE`**(Production이 있을 때, 같은 검증 구간). 비유한 값 거부 | 소영·동찬 |
| 드리프트 (제안) | 최근 21건 RMSE > 10 | 정답이 확보된 최근 28일의 **1주차 RMSE > 같은 기간 naive RMSE**(1주차 정답은 7일 뒤 확보) | 동찬 |
| fine-tuning | 최근 30일, 80:20 분할 | **저장된 전체 데이터의 최근 627행**으로 호출(업로드분만으로는 부족할 수 있음). 최근 365일 학습(Production warm start, scaler 재fit 금지), 학습에 쓰지 않은 최근 90일 검증, 게이트 동일. 반환 `{promoted, rmse, naive_rmse, version?, status}`, `status` = `promoted`·`gate_failed`·`no_production`, 627행 미만은 `ValueError("insufficient_data")` | 소영 → 동찬 |
| 배치·업로드 최소 행 | 41행, `rows[i:i+20]` | **175행**(입력 120 + 짝 28건 − 1 + 4주 정답 28), `rows[i:i+120]` 예측 → 다음 1~4주 실제 평균과 비교, 최근 28건으로 드리프트 판정. AIOps에 넘기는 짝(제안): `{date, predicted[4], actual[4], naive}`, naive = 마지막 입력일 가격 | 준형·동찬 |

게이트 수치 근거
- 50원: 화물 안전운임제는 3개월 평균 경유가가 ±50원 이상 변하면 운임을 다시 고시합니다. 운송업계가 결정을 바꾸는 단위이므로 1주 예측 오차의 업무 허용 한도로 씁니다. 10원은 다음날 휘발유(naive 3~8원) 기준이라 주간 예측에 맞지 않습니다(2026 검증 구간 1주차 naive 34.6원).
- naive 비교: 유가 예측 연구의 표준 평가(무변화 예측 대비 오차 비율 < 1).
- Production 비교: 새 모델이 현재 서비스 모델보다 나쁠 때 교체하지 않습니다.

## v1 휘발유 다음날 예측 (현재 코드)

### 데이터

정규화 CSV와 입력 피처 순서는 다음과 같습니다. 오피넷 원본 파일은 데이터 담당자가 이 형식으로 변환합니다.

| 컬럼 | 의미 | 검증 |
|---|---|---|
| `date` | 기준일, ISO 날짜 | 하루 간격 오름차순, 중복·누락 불가 |
| `diesel_price` | 전국 평균 보통휘발유 가격, 원/L | 양수 |
| `singapore_diesel_price` | 두바이유 현물(오피넷), USD/barrel; D-1일까지 공개된 최근 거래일 값 | 양수 |
| `usd_krw` | 원/미국달러 매매기준율(ECOS 731Y001), KRW/USD; D일까지 고시된 값 | 양수 |
| `tax_or_supply_feature` | 휘발유 유류세 인하율 %, D일 시행값 (공급 차질은 피처 제외) | 유한 실수 |

출처·시차·결측 처리 상세는 [`data/README.md`](../data/README.md)를 따릅니다.

합성 예제의 정책 피처 0은 실제 관측값이 아닙니다. 추가 지표의 출처·라이선스·발표 시각과
휴일 결측 처리 정책을 확정하기 전에는 실측 성능을 주장하지 않습니다. 미래 값으로 보간하지 않습니다.
API sequence는 가장 오래된 날부터 최근 날까지 20개이며 날짜는 생략합니다.
학습·추론 모두 `FEATURE_COLUMNS` 순서로 `(N,20,4)`를 구성하고 마지막 입력 다음날 가격을 예측합니다.
CSV 업로드 최소 41행은 시뮬레이션 21건 확보 기준이며 충분한 학습량을 의미하지 않습니다.

### API

| Method | URL | 요청 / 응답 | 뼈대 상태 |
|---|---|---|---|
| GET | `/` | 원본 기반 대시보드 | 동작 |
| GET | `/health` | `status`, `model_loaded`, `model_version`, `model_source`, `loading_mode` | `status`는 프로세스 생존; readiness는 `model_loaded` (lazy는 첫 예측 전 false) |
| POST | `/data/upload` | multipart CSV → `filename`, `rows` | 검증·저장, 오류 400 |
| GET | `/data/status` | 데이터 기간·행 수·`min_price`·`max_price` | 데이터 없으면 `exists:false` |
| POST | `/predict` | `sequence` → `predicted_price`, `model_version` | 모델 필요, 입력 422, 미준비 503 |
| POST | `/predict/batch-test` | `rows` 21개 이상 → `predictions`, `drift_check` | TODO, 현재 501 |
| GET | `/logs` | 로그 목록 | 원본 재사용 |
| GET | `/logs/{filename}` | `name`, `content` | 원본 재사용, 없는 파일 404 |

전체 예측 요청 예시는 [`examples/predict.json`](../examples/predict.json).
배치의 `rows`는 DailyPoint 전체 피처를 포함합니다. 원본 `prices` 전용 계약은 사용하지 않습니다.
배치에서 `rows[i:i+20]`으로 예측하고 `rows[i+20].diesel_price`를 실제값으로 사용합니다.
응답 `model_version`은 로컬 `v1-local`; MLflow 구현 후 `production:<실제 등록 버전>`으로 식별합니다.

### 모델·운영

- 배포 게이트: 원/L 기준 `RMSE <= 10 AND RMSE < naive_rmse`. 두 지표는 같은 시간순 검증 타깃으로 계산하며 비유한 값은 거부합니다.
- naive는 직전 날 가격. scaler는 baseline 학습 구간에만 fit, 이후 transform만 합니다.
- `fine_tune(rows)`는 Production 가중치에서 warm start하고 `{promoted, rmse, naive_rmse, version?}`를 반환합니다.
- 원본 MLflow `Production` stage와 URI를 유지했습니다. alias 전환은 모델·서빙 담당이 함께 변경합니다.
- 재학습 정책 목표: 최근 30일 학습 대상 + 선행 20일 입력 문맥. **이 50행 전체를 학습과 성능 검증에 중복 사용하지 않습니다.** 독립 시간순 검증 구간/최소 표본 수를 모델 담당자가 추가하고 확정합니다. 현재 원본 `_prepare`의 80:20 분할은 최종 정책이 아닙니다.
- 드리프트 초기값: 정답이 확보된 최근 21건의 RMSE > 10원/L. 검증을 거쳐 조정할 초기 가설입니다.
- 실제 다음날 가격이 도착하기 전에는 RMSE를 계산할 수 없습니다. 배치 시뮬레이션과 실운영 실제값 연결을 구분합니다.
- 데이터 부족은 `insufficient_data`, 정상은 `ok`; 재학습은 실행 상태와 `promoted`를 별도 표시하도록 구현합니다.
- 감지 → `[WARN]` → `[INFO] retrain triggered` → 게이트 재검증 → 성공 시 `[OK]`와 실제 버전. 실패 시 기존 Production 유지.
- 중복 트리거 방지, 실패 로그, 성공 후 모델 캐시 교체와 예측 윈도우 초기화는 AIOps·서빙 공동 완료 항목입니다.
- 알림은 `logs/aiops.log [WARN]` + 운영자 알림이 필수입니다. 운영자 알림 채널(예: 웹훅/메일)과 수신 대상은 AIOps 담당이 확정하고 설정 가능한 어댑터로 구현합니다. 전송 실패 로그와 중복 억제를 검증합니다. 현재 운영자 전송은 미구현입니다.

## Docker 실행·영속화

단일 컨테이너(`serving_app/docker-compose.yml`, 포트 8000). 런타임 데이터는 이름 있는 볼륨에 둡니다.

| 볼륨 | 컨테이너 경로 | 내용 |
|---|---|---|
| `uploads` | `/app/data/uploads` | 업로드 CSV (학습은 최신 파일 사용) |
| `models` | `/app/serving_app/models` | 로컬 모델 `.keras`, `scaler.pkl` |
| `logs` | `/app/logs` | `aiops.log` |
| `mlflow-db` | `/app/runtime` | `mlflow.db` (`MLFLOW_TRACKING_URI=sqlite:////app/runtime/mlflow.db`) |
| `mlruns` | `/app/mlruns` | MLflow 아티팩트(등록 모델 가중치) |

- `restart`·`down`·재빌드 후에도 유지되고, `down -v`로만 삭제됩니다. `restart: unless-stopped`.
- 호스트의 `mlflow.db`·`mlruns`는 공유하지 않습니다. 아티팩트 경로가 호스트 절대경로로 기록되어 컨테이너에서 찾을 수 없으므로 컨테이너 안에서 학습·등록합니다.
- 모델이 없으면 서버는 뜨고 `/predict`만 503입니다(기본 lazy). eager는 시작 시 로드 실패가 바로 드러나지만, 모델이 준비된 뒤 전환합니다.

```bash
make docker                                   # 빌드 + 실행 (lazy/local)
docker compose -f serving_app/docker-compose.yml exec serving-app python scripts/train_baseline_v1.py      # local 모델
docker compose -f serving_app/docker-compose.yml exec serving-app python serving_app/train_and_register.py # MLflow 등록
MODEL_SOURCE=mlflow docker compose -f serving_app/docker-compose.yml up -d    # MLflow Production 서빙
```
