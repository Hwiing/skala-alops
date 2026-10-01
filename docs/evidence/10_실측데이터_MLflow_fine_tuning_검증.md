# 실측 데이터·MLflow fine-tuning 검증

담당: 데이터·도메인 (`@yunanana`) · 연관 이슈: [#17](https://github.com/Hwiing/skala-alops/issues/17)

## 목적

재학습 트리거 이슈의 남은 확인 항목인 “실데이터·실제 MLflow로 fine-tuning 검증”을 기록합니다. 테스트 대역이나 합성 데이터 대신 팀에서 선택한 2008년 확장 CSV와 MLflow Model Registry를 사용했습니다.

## 데이터와 실행 환경

| 항목 | 값 |
|---|---|
| 데이터 | `data/processed/diesel_features_2008_spliced.csv` |
| 행 수·기간 | 6,743행, 2008-04-15~2026-09-30 |
| CSV SHA-256 | `a77c19369d858300499f9744ad37a9827b40a1f337085a35a99d76d3990de2a9` |
| 전달 ZIP SHA-256 | `1d6d0caa1bf96cc9e2312c402448356556bff49c3c73d4b44575f8a3dd785ae6` |
| Python / MLflow / TensorFlow | 3.11.16 / 3.16.0 / 2.21.0 |
| Tracking store | 로컬 SQLite (`mlflow-real.db`), MLflow 서버 없이 실제 MLflow API 사용 |
| 데이터 성격 | 합성 예제가 아님. 국내 경유·환율은 실측. 앞 1,694행의 국제가격은 경유(0.05%) 실측에 평균 규격 스프레드 1.479667 USD/bbl를 더한 접합 추정값이며 provenance에 표시됨 |

CSV와 원본 ZIP, MLflow DB는 데이터 이용 조건과 파일 크기 때문에 저장소에 포함하지 않습니다. CSV는 팀 Slack으로 인계한 ZIP에서 꺼내 지문을 확인했습니다.

## 재현 명령

저장소 루트에서 의존성을 설치하고, 전달 ZIP의 CSV를 `data/processed/` 아래에 놓은 다음 순서대로 실행합니다. 별도의 새 로컬 SQLite DB를 사용합니다.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

MLFLOW_TRACKING_URI=sqlite:///mlflow-real.db \
  .venv/bin/python serving_app/diesel_registry.py \
  --csv data/processed/diesel_features_2008_spliced.csv

MLFLOW_TRACKING_URI=sqlite:///mlflow-real.db \
  .venv/bin/python serving_app/diesel_registry.py \
  --csv data/processed/diesel_features_2008_spliced.csv --fine-tune
```

첫 번째 실행은 실제 base 모델을 학습·평가해 게이트를 통과하면 Production과 `champion`에 등록합니다. 두 번째 실행은 등록된 Production 모델에서 warm start해 `fine_tune()`을 실행하고, 최근 627행만 사용합니다.

## 실제 실행 결과

### Base 모델 등록

| 항목 | 결과 |
|---|---|
| MLflow run | `4ec418b640ae4e9a902f8216fdead433` (`diesel-base-train`) |
| 평가 기간 | 2025-09-03~2026-09-02, 365개 검증 기준일 |
| 학습 | 3 seeds `[42, 7, 2026]`, 각 40 epochs |
| 모델 주별 RMSE (원/L) | `[27.56, 58.25, 73.95, 81.32]` |
| naive 주별 RMSE (원/L) | `[28.78, 59.56, 76.27, 89.82]` |
| 결과 | 게이트 통과, `DieselPricePredictor` v1을 Production 및 `champion`으로 등록 |

### Fine tuning

| 항목 | 결과 |
|---|---|
| MLflow run | `ea76d08498974cdc85530f95152e6d77` (`diesel-fine-tune`) |
| 사용한 데이터 | 최근 627행, 2025-01-12~2026-09-30 |
| 학습 정답 기간 | 2025-05-08~2026-05-07, 365개 기준일 |
| 독립 검증 기간 | 2026-06-05~2026-09-02, 90개 기준일 |
| Fine-tuned 모델 주별 RMSE (원/L) | `[16.09, 14.91, 25.70, 41.00]` |
| naive 주별 RMSE (원/L) | `[15.50, 37.67, 54.79, 68.45]` |
| 같은 기간 Production RMSE (원/L) | `[16.10, 14.67, 25.21, 40.68]` |
| 결과 | `gate_failed`; 신규 버전 미등록, 기존 Production/`champion` v1 유지 |

게이트 실패 사유는 두 가지입니다.

1. 1주차 RMSE `16.09`가 naive `15.50`보다 높음
2. 4주 평균 RMSE `24.43`이 기존 Production `24.16`보다 높음

이는 fine-tuning 실행 오류가 아닙니다. 실제 학습·평가와 MLflow 기록까지 완료됐고, 새 모델이 기준을 만족하지 못해 기존 서비스 모델을 보존한 결과입니다. MLflow Registry에 남은 모델 버전은 Production v1 하나이며 `champion` alias도 v1을 가리켰습니다.

## 범위와 한계

- 실제 MLflow run, 모델 등록, Production warm start, 627행 fine-tuning, 독립 검증, 게이트 판정, 실패 시 Production 보존을 확인했습니다.
- 결과는 해당 CSV 지문·설정·검증 기간에 대한 단일 실험입니다. 실시간 서비스 교체, Docker 실행, 운영자 알림 전체 흐름의 검증 결과로 확대 해석하지 않습니다.
- 로컬 MLflow DB는 재현 실행 시 새로 생성되며 Git에 커밋하지 않습니다. 위 run ID는 당시 로컬 DB 기록을 식별합니다.
