# Oilgorithm

## 경유 가격 예측 및 AIOps 파이프라인

> **운송·물류회사가 향후 유류비를 미리 예측하고, 시장 환경 변화에도 지속적으로 신뢰할 수 있는 예측 서비스를 사용할 수 있도록 하는 경유 가격 예측·운영 시스템**

Oilgorithm은 **다음 1~4주 전국 평균 자동차용 경유 가격(원/L)**을 예측하고, 모델 성능 저하가 발생하면 이를 자동으로 감지하여 **재학습 → 성능 검증 → Production 교체**까지 수행하는 AI 서비스입니다.

---

# 1. Pain Point — 왜 필요한가?

운송·물류회사의 비용 구조에서 유류비는 중요한 변동비 중 하나입니다.

경유 가격이 단기간에 크게 상승하면 운송 원가가 증가하지만, 실제 가격 변화를 확인한 이후 대응한다면 배차 계획, 운임 산정, 예산 관리가 이미 늦어질 수 있습니다.

따라서 운영 담당자에게 필요한 것은 단순히 **현재 경유 가격을 확인하는 것**이 아니라,

> **향후 수 주간 경유 가격이 어떻게 움직일지를 미리 파악하여 유류비 계획에 반영하는 것**

입니다.

또한 AI 모델을 한 번 학습했다고 해서 계속 같은 성능을 유지하는 것도 아닙니다.

국제 경유 가격, 환율, 세금 정책, 공급 상황 등이 변화하면 학습 당시와 데이터 패턴이 달라져 예측 오차가 증가할 수 있습니다.

일반 소프트웨어 장애와 달리 AI 모델은 **API가 정상적으로 200 응답을 반환하면서도 잘못된 예측을 계속 제공할 수 있습니다.**


---

# 2. AI 솔루션 및 운영 목표

## 2.1 예측 서비스

모델은 최근 데이터를 기반으로 다음 **1·2·3·4주 전국 평균 자동차용 경유 가격**을 예측합니다.

### 입력 데이터

주요 데이터는 다음과 같습니다.

| 데이터           | 역할                 |
| ------------- | ------------------ |
| 국내 자동차용 경유 가격 | 예측 대상 및 핵심 시계열     |
| 싱가포르 경유 가격    | 국제 석유제품 가격 변화 반영   |
| 원/달러 환율       | 수입 원가 변화 반영        |
| 경유 유류세        | 정책에 따른 가격 구조 변화 반영 |

API는 날짜를 포함한 **최근 120일 데이터**를 입력받으며, 내부 모델은 전처리 후 **28일 × 8개 피처** 시퀀스를 사용합니다.

---

## 2.2 서비스 운영 목표

모델 정확도만 높이는 것이 아니라 **운영 가능한 모델인지**를 함께 판단합니다.

### 배포 기준

새 모델은 다음 조건을 모두 만족해야 Production 후보가 됩니다.

```text
1주 RMSE ≤ 50원/L

AND

1·2·3·4주 예측 RMSE가
각각 Naive 모델보다 우수

AND

4주 평균 RMSE가
현재 Production 모델보다 우수
```

---

# 3. 운영 설계

## 3.1 모델 학습 및 배포

모델 학습 결과는 MLflow에 기록하고, Gate를 통과한 모델만 Registry의 `champion`으로 승격합니다.

```text
데이터 준비
    ↓
모델 학습
    ↓
MLflow Tracking
    ↓
성능 평가
    ↓
Deployment Gate
    ↓
MLflow Registry
    ↓
champion 승격
    ↓
FastAPI Serving
```

---

## 3.2 Drift 판단

운영 중에는 정답이 확보된 **최근 28개 기준일**의 1주 예측 결과를 이용하여 성능을 평가합니다.

Drift 기준은 다음과 같습니다.

```text
최근 28개 기준일의 1주 RMSE
>
max(
    같은 기간 Naive RMSE,
    10원/L
)
```

즉 단순히 RMSE가 조금 증가했다고 바로 재학습하는 것이 아니라,

* 현재 모델이 Naive보다 나빠졌는지
* 최소 허용 오차 수준도 넘어섰는지

를 함께 확인합니다.

---

## 3.3 자동 재학습

Drift가 감지되면 새로운 모델을 처음부터 학습하지 않고 현재 Production 모델을 기반으로 **Warm Start Fine-tuning**을 수행합니다.

```text
Drift 감지
   ↓
운영자 경고
   ↓
최근 데이터 확보
   ↓
Production 모델 Warm Start
   ↓
Fine-tuning
   ↓
독립 검증
   ↓
배포 Gate 재검증
```

재학습에는 최소 **627행**이 필요하며,

* 최근 365일: 학습
* 독립 90일: 검증

구조를 사용합니다.

새 모델이 Gate를 통과하지 못하면 **현재 Production 모델을 그대로 유지**합니다.

---

## 3.4 운영자 알림

AIOps 이벤트는 기본적으로

```text
logs/alerts.jsonl
```

에 기록합니다.

대표 상태는 다음과 같습니다.

```text
[WARN] drift detected
        ↓
[INFO] retrain triggered
        ↓
[OK] new model promoted
```

---

# 4. 전체 아키텍처

```text
[경유 데이터 CSV]
 자동차용 경유
 싱가포르 경유
 환율
 유류세
        ↓
[데이터 검증·전처리]
 날짜 / 결측 / 연속성 / 값 범위
        ↓
[Feature Engineering]
        ↓
[LSTM 학습]
        ↓
[MLflow Tracking]
        ↓
[Deployment Gate]
        ↓
[MLflow Registry]
        ↓
[champion]
        ↓
[FastAPI Serving]
   ├─ /predict
   ├─ /health
   ├─ /data/upload
   └─ /predict/batch-test
        ↓
[Docker Container]
        ↓
[Prediction 기록]
        ↓
[최근 28개 기준일 RMSE]
        ↓
[Drift 판단]
        ↓
[logs/alerts.jsonl]
        ↓
[Warm Start Fine-tuning]
        ↓
[독립 Validation]
        ↓
[Gate 재검증]
      ↙        ↘
   통과          실패
    ↓             ↓
champion 승격   기존 모델 유지
    ↓
Serving Model 교체
```


---

# 5. API 명세

## 주요 Endpoint

| Method | Endpoint              | 역할                         |
| ------ | --------------------- | -------------------------- |
| `GET`  | `/health`             | 서버 및 모델 상태 확인              |
| `POST` | `/predict`            | 향후 1~4주 경유 가격 예측           |
| `POST` | `/predict/batch-test` | 과거 데이터를 이용한 배치 평가·Drift 검사 |
| `POST` | `/data/upload`        | 경유 학습 데이터 CSV 업로드          |
| `GET`  | `/data/status`        | 현재 업로드 데이터 상태 확인           |
| `GET`  | `/logs/{filename}`    | 운영·AIOps 로그 조회             |

Swagger:

```text
http://localhost:8000/docs
```

### Prediction 예시

```bash
curl \
  -H 'Content-Type: application/json' \
  --data-binary @examples/predict.json \
  http://localhost:8000/predict
```

### Batch Test 예시

```bash
curl \
  -H 'Content-Type: application/json' \
  --data-binary @examples/batch-test.json \
  http://localhost:8000/predict/batch-test
```

---

# 6. 실행 방법

## 6.1 개발 환경

Python 3.11을 기준으로 하며 모든 명령은 저장소 루트에서 실행합니다.

```bash
make setup-dev
make lint test
make run
```

실행 후:

```text
Dashboard
http://localhost:8000/

Swagger
http://localhost:8000/docs
```

---

## 6.2 데이터 업로드

```bash
curl \
  -F 'file=@data/processed/diesel_features_2008_spliced.csv' \
  http://localhost:8000/data/upload
```

상태 확인:

```bash
curl http://localhost:8000/data/status
```

`data/sample_diesel_prices.csv`는 **API 계약 확인용 120일 합성 데이터**이며 실제 모델 학습 성능 증빙에는 사용하지 않습니다.

---

## 6.3 모델 학습

전체 의존성 설치:

```bash
make setup
```

Baseline 학습:

```bash
.venv/bin/python scripts/train_diesel_baseline.py \
  --csv data/processed/diesel_features_2008_spliced.csv
```

MLflow 학습 및 Registry 등록:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
.venv/bin/python serving_app/diesel_registry.py \
  --csv data/processed/diesel_features_2008_spliced.csv
```

운영 모델 서빙:

```bash
MODEL_SOURCE=mlflow \
LOADING_MODE=eager \
make run
```

MLflow는 별도 서버 없이 **로컬 SQLite**를 사용합니다.

---

## 6.4 Docker

단일 컨테이너 환경으로 실행합니다.

```bash
make docker
```

컨테이너에서도 동일하게

```text
/predict
/health
/data/upload
/predict/batch-test
```

Endpoint가 동작하는지 확인합니다.

---

# 7. 동작 검증


### ① 정상 예측

```text
데이터 입력
→ /predict
→ 1~4주 가격 예측
```

### ② 모델 배포

```text
모델 학습
→ 성능 평가
→ Gate 통과
→ MLflow champion 승격
→ 실제 Serving 모델 교체
```

### ③ Drift 발생

```text
최근 예측 성능 악화
→ Drift 판단
→ WARN
```

### ④ 자동 복구

```text
Drift
→ Fine-tuning
→ Validation
→ Gate
→ champion 승격
→ Serving 교체
```

### ⑤ 실패 안전장치

```text
새 모델 Gate 실패
→ 기존 Production 유지
```

---

# 8. 프로젝트 구조

```text
.
├── data/
│   ├── contracts.py
│   ├── README.md
│   └── processed/
│
├── serving_app/
│   ├── model_loader.py
│   ├── schemas.py
│   ├── diesel_model.py
│   ├── diesel_registry.py
│   ├── routers/
│   └── monitoring/
│
├── scripts/
│   ├── train_diesel_baseline.py
│   └── simulate_drift.py
│
├── docs/
│   ├── contracts.md
│   ├── source-mapping.md
│   ├── team-plan.md
│   └── evidence/
│
└── examples/
```

세부 입력·출력 계약은 `docs/contracts.md`를 기준으로 합니다.

---

# 9. 팀 역할

| 역할             | 담당                                              | 주요 범위                    |
| -------------- | ----------------------------------------------- | ------------------------ |
| 데이터·도메인        | [kwon yuna](https://github.com/yunanana)        | `data/`, CSV 계약          |
| 모델·MLflow      | [bookschooler](https://github.com/bookschooler) | Baseline, 모델, Registry   |
| Serving·Docker | [leejunhyeong](https://github.com/hootbee)      | FastAPI, Loader, Docker  |
| AIOps          | [kchanis1223](https://github.com/kchanis1223)   | Drift, Retraining, Alert |
| 통합·PM/발표       | [Hwiing](https://github.com/Hwiing)             | 통합 검증, 문서, 발표            |

각 담당 기능은 독립적인 결과물이 아니라 최종적으로 다음 하나의 파이프라인으로 연결됩니다.

```text
Data
→ Model
→ Registry
→ Serving
→ Monitoring
→ Drift
→ Retraining
→ Redeployment
```
