# SKALA 경유 주간 가격 예측 · AIOps

운송·물류회사의 유류비 계획을 돕기 위해 **다음 1~4주 전국 평균 자동차용경유 가격(원/L)**을 예측합니다.
공통 계약은 [docs/contracts.md](docs/contracts.md), 상수·CSV 검증은 [data/contracts.py](data/contracts.py),
요청·응답·AIOps 결과 타입은 [serving_app/schemas.py](serving_app/schemas.py)를 기준으로 합니다.

## 팀 확정 도메인 매핑

| 항목 | 확정 내용 |
|---|---|
| 예측 대상 | 다음 1·2·3·4주 경유 평균가 |
| API 입력 | 날짜 포함 최근 120일 × 경유 4피처, 하루 간격 |
| 모델 | 내부 28일 × 8피처 LSTM·scaler·정책 규칙을 포함한 MLflow pyfunc |
| 데이터 공급 | 자동차용경유·싱가포르 경유·ECOS 환율·경유 유류세 |
| 배포 게이트 | 1주 RMSE ≤ 50, 4주 각각 naive보다 우수, 4주 평균 RMSE ≤ Production |
| 드리프트 계약 | 정답이 확보된 최근 28개 기준일의 1주 RMSE > 같은 기간 naive RMSE |
| 재학습 | 최근 627행, Production warm start·365일 학습·독립 90일 검증 |
| 이해관계자 | 운송·물류회사 |

## 현재 상태

- 날짜·양수/유한값·연속성 검증, 120일 예측 API, 175행 배치와 주간 정답 연결, pyfunc 로더를 구현했습니다.
- Registry 승격 버전과 실제 서빙 버전이 일치할 때만 교체 성공으로 표시하고 예측 기록을 비웁니다.
- 모델 미준비는 503입니다. 드리프트 계산·자동 재학습 트리거·운영자 알림은 아직 TODO이며 배치가 해당 코드에 도달하면 501입니다.
- `data/sample_diesel_prices.csv`는 **120일 합성 예제**입니다. 175행 업로드나 실제 학습 성능 증빙으로 쓰지 않습니다.
- `examples/predict.json`, `examples/batch-test.json`도 API 계약 확인용 합성 요청입니다.
- MLflow는 별도 서버 없이 로컬 SQLite, Docker는 단일 컨테이너입니다. 영속 볼륨·UI 후속 PR 통합 검증은 별도입니다.

## 실행

Python 3.11, uv, make를 사용하며 모든 명령은 저장소 루트에서 실행합니다.

```bash
make setup-dev  # API·계약 테스트, TensorFlow/MLflow 제외
make lint test
make run
```

대시보드 <http://localhost:8000/>, Swagger <http://localhost:8000/docs>.

```bash
curl http://localhost:8000/health
curl -H 'Content-Type: application/json' --data-binary @examples/predict.json http://localhost:8000/predict
curl -H 'Content-Type: application/json' --data-binary @examples/batch-test.json http://localhost:8000/predict/batch-test
# 최소 175행인 정규화 경유 CSV 업로드 (파일 경로는 실제 전달본에 맞춤)
curl -F 'file=@data/processed/diesel_features_2008_spliced.csv' http://localhost:8000/data/upload
curl http://localhost:8000/data/status
```

실제 경유 모델을 학습·등록하려면 전체 의존성과 [데이터 재생성/인계](data/README.md)에서 설명한
원본·가공 파일이 필요합니다. CSV 업로드가 모델 학습을 자동 실행하지는 않습니다.

```bash
make setup
.venv/bin/python scripts/train_diesel_baseline.py --csv data/processed/diesel_features_2008_spliced.csv
MLFLOW_TRACKING_URI=sqlite:///mlflow.db .venv/bin/python serving_app/diesel_registry.py --csv data/processed/diesel_features_2008_spliced.csv
# 게이트 통과로 champion이 준비된 뒤 전환
MODEL_SOURCE=mlflow LOADING_MODE=eager make run
```

`.env.example`은 설정 예시이며 앱이 자동으로 읽지 않습니다. 셸 환경변수로 전달하세요.
로컬 pyfunc는 `serving_app/models/diesel_pyfunc`, Registry 버전은 `champion:<번호>`로 식별합니다.
학습 실패나 게이트 미통과 시 Production이 없을 수 있습니다. 실측 학습·실제 버전 전환·Docker 실행
증빙은 공통 계약 테스트와 별도로 확인합니다. `make docker`로 단일 컨테이너를 실행할 수 있습니다.

## 구조와 역할

| 담당 | GitHub | 주 작업 경로 |
|---|---|---|
| 데이터·도메인 | [kwon yuna](https://github.com/yunanana) | `data/`, CSV 계약 |
| 모델·MLflow | [bookschooler](https://github.com/bookschooler) | `scripts/train_diesel_baseline.py`, `serving_app/diesel_model.py`, `diesel_registry.py` |
| 서빙·Docker | [leejunhyeong](https://github.com/hootbee) | `serving_app/model_loader.py`, `schemas.py`, `routers/`, Docker |
| AIOps | [kchanis1223](https://github.com/kchanis1223) | `serving_app/monitoring/`, `scripts/simulate_drift.py` |
| 통합·PM/발표 | [Hwiing](https://github.com/Hwiing) | `docs/`, CI, 통합 검증·발표 |

공통 규격은 [계약](docs/contracts.md), 원본 재사용 내역은 [출처와 변경](docs/source-mapping.md),
통합 순서와 발표 항목은 [협업 계획](docs/team-plan.md)을 참고하세요.
브랜치는 `feat/<issue-number>-<topic>`, PR에는 이슈 링크와 실제 검증 결과를 첨부합니다.
개인 실습 폴더 `.local-practice/`, 업로드 데이터, 모델, SQLite, 로그는 커밋하지 않습니다.
