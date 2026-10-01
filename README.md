# SKALA 휘발유 가격 예측 · AIOps

운송·물류회사의 유류비 계획을 돕기 위해 **다음날 전국 평균 보통휘발유 가격(원/L)**을 예측합니다.
개인 실습 `project/` 코드를 복사해 팀 도메인으로 개조한 협업용 뼈대입니다.

## 팀 확정 도메인 매핑

| 항목 | 확정 내용 |
|---|---|
| 예측 대상 | 다음날 전국 평균 휘발유 가격 |
| 입력 시퀀스 | 최근 20일 가격, 국제유가, 환율, 유류세/공급 관련 지표 |
| 데이터 공급 | 오피넷 CSV 업로드; 추가 데이터 후보는 국제유가·환율 |
| 배포 게이트 | RMSE ≤ 10원/L AND naive(어제값) 모델보다 RMSE가 낮을 것 |
| 드리프트 신호 | 국제유가 급등락, 환율 급변, 유류세 정책 변경, 공급 차질 |
| 재학습 | 최근 30일 데이터 fine-tuning |
| 알림 | aiops.log [WARN] + 운영자 알림 (채널·수신 대상 확정 필요) |
| 이해관계자 | 운송·물류회사 |

## 현재 상태

- 원본 FastAPI, 대시보드, CSV 업로드, 로그 조회, Lazy/Eager 로더, LSTM 학습, MLflow 등록, Docker, AIOps 모듈 구조를 재사용합니다.
- 최근 20일 × 4피처, 다음날 가격 출력, 원/L RMSE와 naive 비교 게이트를 연결했습니다.
- 원본 핵심 TODO 5개는 담당자 구현 대상으로 남겼습니다. `/predict/batch-test`는 현재 **501**, 모델 준비 전 `/predict`는 **503**입니다. 정상 예측이나 자동 재학습이 완성된 상태가 아닙니다.
- `data/sample_diesel_prices.csv`는 **120일 합성 예제**입니다. 오피넷 실측 데이터 또는 성능 증빙으로 사용하지 않습니다.
- MLflow는 별도 서버 없이 로컬 SQLite, Docker는 단일 컨테이너입니다.

## 실행

Python 3.11, uv, make를 사용합니다. 모든 명령은 저장소 루트에서 실행합니다.

```bash
make setup-dev  # API와 테스트만 설치; TensorFlow/MLflow 제외
make lint test
make run
```

대시보드 <http://localhost:8000/>, Swagger <http://localhost:8000/docs>.

```bash
curl http://localhost:8000/health
curl -F 'file=@data/sample_diesel_prices.csv' http://localhost:8000/data/upload
curl http://localhost:8000/data/status
```

로컬 모델까지 실행하려면 전체 의존성을 설치하고 CSV 업로드 후 학습합니다.

```bash
make setup
.venv/bin/python scripts/train_baseline_v1.py
curl -H 'Content-Type: application/json' --data-binary @examples/predict.json http://localhost:8000/predict
MLFLOW_TRACKING_URI=sqlite:///mlflow.db .venv/bin/python serving_app/train_and_register.py
# _load_from_mlflow TODO 및 게이트 통과 확인 후:
MODEL_SOURCE=mlflow LOADING_MODE=eager make run
```

`.env.example`은 설정 예시이며 앱이 자동으로 읽지 않습니다. 셸 환경변수로 전달하세요.
합성 데이터로 게이트 통과를 보장하지 않습니다. 미통과 시 Production이 없을 수 있습니다.
학습/등록은 모델 담당자가 실제 데이터와 시간순 검증 정책으로 검증해야 합니다.

```bash
make docker
# 원본의 빌드 시 학습 흐름은 선택적으로 보존되어 있습니다 (무거운 전체 학습).
# 담당자 구현/검증 완료 후 사용:
# docker build --build-arg TRAIN_ON_BUILD=1 -f serving_app/Dockerfile -t gasoline-serving .
```

기본 이미지는 모델 없는 lazy/local 상태로 시작합니다. MLflow 로더가 미구현이므로
Production/eager 전환과 영속 볼륨 구성은 서빙 담당 이슈에서 완료합니다.

## 구조와 역할

| 담당 | GitHub | 주 작업 경로 |
|---|---|---|
| 데이터·도메인 | [kwon yuna](https://github.com/yunanana) | `data/`, CSV 계약 |
| 모델·MLflow | [bookschooler](https://github.com/bookschooler) | `scripts/train_baseline_v1.py`, `serving_app/lstm_model.py`, `train_and_register.py` |
| 서빙·Docker | [leejunhyeong](https://github.com/hootbee) | `serving_app/model_loader.py`, `schemas.py`, `routers/`, Docker |
| AIOps | [kchanis1223](https://github.com/kchanis1223) | `serving_app/monitoring/`, `scripts/simulate_drift.py` |
| 통합·PM/발표 | [Hwiing](https://github.com/Hwiing) | `docs/`, CI, 통합 검증·발표 |

공통 규격은 [계약](docs/contracts.md), 원본 재사용 내역은 [출처와 변경](docs/source-mapping.md),
통합 순서와 발표 항목은 [협업 계획](docs/team-plan.md)을 참고하세요.
브랜치는 `feat/<issue-number>-<topic>`, PR에는 이슈 링크와 실제 검증 결과를 첨부합니다.
개인 실습 폴더 `.local-practice/`, 업로드 데이터, 모델, SQLite, 로그는 커밋하지 않습니다.
