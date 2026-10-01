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
| 최종 재실행 코드 | main `bbeab1eafce6b3ffeea89350d121ed20a4b1ab12` (PR #51의 `RegistryVersion` 수정 포함) |
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
| 실행 코드 | PR #47 작업본 `23a1de740b8cda57abaa8806132109ce00828ea4` |
| 평가 기간 | 2025-09-03~2026-09-02, 365개 검증 기준일 |
| 학습 | 3 seeds `[42, 7, 2026]`, 각 40 epochs |
| 모델 주별 RMSE (원/L) | `[27.56, 58.25, 73.95, 81.32]` |
| naive 주별 RMSE (원/L) | `[28.78, 59.56, 76.27, 89.82]` |
| 결과 | 게이트 통과, `DieselPricePredictor` v1을 Production 및 `champion`으로 등록 |

### 초기 Fine tuning 기록

| 항목 | 결과 |
|---|---|
| MLflow run | `ea76d08498974cdc85530f95152e6d77` (`diesel-fine-tune`) |
| 사용한 데이터 | 최근 627행, 2025-01-12~2026-09-30 |
| 학습 정답 기간 | 2025-05-08~2026-05-07, 365개 기준일 |
| 독립 검증 기간 | 2026-06-05~2026-09-02, 90개 기준일 |
| Fine-tuned 모델 주별 RMSE (원/L) | `[16.09, 14.91, 25.70, 41.00]` |
| naive 주별 RMSE (원/L) | `[15.50, 37.67, 54.79, 68.45]` |
| 같은 기간 Production RMSE (원/L) | `[16.10, 14.67, 25.21, 40.68]` |
| 결과 | 학습·평가 후 게이트 실패 결과 생성. 당시 실행 경로는 최신 `FineTuneResult` 반환 계약 검증까지 입증하지 못함 |

초기 기록의 게이트 실패 사유는 두 가지입니다.

1. 1주차 RMSE `16.09`가 naive `15.50`보다 높음
2. 4주 평균 RMSE `24.43`이 기존 Production `24.16`보다 높음

이 기록만으로 함수가 예외 없이 응답했다고 판단할 수 없습니다. PR #50 리뷰에서 MLflow의 정수 버전과 응답 스키마의 문자열 계약 충돌이 발견됐고, PR #51에서 공통 `RegistryVersion` 스키마로 수정됐습니다.

### 최신 main에서 실측 재실행

main `bbeab1eafce6b3ffeea89350d121ed20a4b1ab12`의 **전체 소스**를 사용했습니다. 위 SHA-256의 실측 CSV와 앞서 등록한 Production v1이 든 로컬 MLflow SQLite의 복사본으로, 위 `--fine-tune` 명령을 재실행했습니다. 버전 정규화는 PR #51의 공통 스키마를 사용합니다.

| 항목 | 결과 |
|---|---|
| 프로세스 종료 코드 | `0` |
| MLflow run | `1b6b7349934340c584f5c7d1181d29f3` (`diesel-fine-tune`) |
| 입력 데이터 | 최근 627행, 2025-01-12~2026-09-30 |
| 학습·독립 검증 기준일 | 학습 365개(2025-05-08~2026-05-07), 검증 90개(2026-06-05~2026-09-02) |
| Fine-tuned RMSE (원/L) | `[16.073334, 14.879276, 25.667145, 40.953282]` |
| naive RMSE (원/L) | `[15.498181, 37.674986, 54.788484, 68.448448]` |
| 같은 기간 Production RMSE (원/L) | `[16.088069, 14.618727, 25.171303, 40.644117]` |
| 반환 | `FineTuneResult` 검증 통과, `gate_failed`, `production_before="1"`, `version=null` |

CLI가 출력한 반환 dict를 같은 값의 JSON 형식으로 적으면 다음과 같습니다.

```json
{
  "rmse": [16.07333398649635, 14.879276235023106, 25.66714472272944, 40.95328171013642],
  "naive_rmse": [15.498181046076377, 37.67498618765987, 54.788484208655404, 68.44844708779122],
  "production_rmse": [16.088069430609732, 14.618727189200508, 25.171302502468095, 40.644116936840476],
  "production_before": "1",
  "passed": false,
  "run_id": "1b6b7349934340c584f5c7d1181d29f3",
  "reasons": [
    "1주차 RMSE 16.07 >= naive 15.50",
    "1~4주 평균 RMSE 24.39 > Production 24.13"
  ],
  "promoted": false,
  "version": null,
  "status": "gate_failed"
}
```

마지막 판정 로그:

```text
[GATE FAILED] 1주차 RMSE 16.07 >= naive 15.50; 1~4주 평균 RMSE 24.39 > Production 24.13 → 등록 안 함. 기존 Production v1 유지
```

반환 후 MLflow Registry를 다시 조회해 Production v1과 `champion` alias v1을 확인했습니다. 학습·MLflow 기록·게이트 판정에 더해, `FineTuneResult` 반환까지 완료된 실행입니다. 모델 성능은 두 게이트를 통과하지 못해 승격되지 않았습니다.

## 범위와 한계

- 실제 MLflow run, 모델 등록, Production warm start, 627행 fine-tuning, 독립 검증, 게이트 판정, `FineTuneResult` 반환, 실패 시 Production 보존을 확인했습니다.
- 결과는 해당 CSV 지문·설정·검증 기간에 대한 단일 실험입니다. 실시간 서비스 교체, Docker 실행, 운영자 알림 전체 흐름의 검증 결과로 확대 해석하지 않습니다.
- 로컬 MLflow DB는 재현 실행 시 새로 생성되며 Git에 커밋하지 않습니다. 위 run ID는 당시 로컬 DB 기록을 식별합니다.
