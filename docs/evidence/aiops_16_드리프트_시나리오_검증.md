# AIOps #16 드리프트 시나리오·운영자 알림 검증

> **합성 데이터 + 스텁 모델 결과입니다. 모델 성능 증빙이 아닙니다.**
> 목적은 감지 → 재학습 → 승격 → 알림 → 서빙 교체 흐름과 판정 기준(1주차 RMSE vs naive)의 동작 확인입니다.
> 실모델 결과는 `data/processed` 경유 데이터와 MLflow champion 모델로 같은 명령을 다시 돌려 교체합니다.

## 조건

- 브랜치: `feat/16-drift-scenarios` (main #46 공통 계약 `diesel-weekly-v2`·#34 서빙 반영 후 재실행)
- 응답은 공개 `DriftCheck` 계약: 표의 1주차/naive RMSE = `drift_rmse`/`drift_naive_rmse`, 조치 = `status`
- 배치: 시나리오당 175행, seed=42, 충격일 125번째 행, 시나리오마다 다른 기간
- 판정: 정답 확보 최근 28건의 1주차 RMSE > 같은 기간 naive(기준일 가격) RMSE → drift
- 스텁 모델(`scripts/aiops_stub_server.py`): 원가(싱가포르가×환율)·세금을 보고 소매가가 하루 15%씩 따라간다고 예측. fine_tune·reload는 가짜(항상 승격·교체 성공)

## 재현 명령

```bash
python scripts/mock_webhook.py --port 9019 &                 # 운영자 알림 수신 (실패 검증: --fail)
STUB_BIAS=0 AIOPS_ALERT_WEBHOOK_URL=http://127.0.0.1:9019 RETRAIN_COOLDOWN_SECONDS=0 \
  python scripts/aiops_stub_server.py &                      # 포트 8011
python scripts/simulate_drift.py --url http://127.0.0.1:8011/predict/batch-test --out logs/scenario_results.md
# 개념 드리프트 흉내: 서버를 STUB_BIAS=40으로 다시 띄우고 같은 명령
```

## 결과 1: 모델이 입력 관계를 제대로 아는 경우 (STUB_BIAS=0)

| 시나리오 | 입력 충격(대조군 대비 최대) | 판정 | 1주차 RMSE | naive RMSE | 조치 |
|---|---|---|---|---|---|
| normal | 충격 없음 | ok | 7.11 | 10.64 | ok |
| oil_shock | 소매 +280원, 싱가포르 +35% | ok | 34.71 | 52.25 | ok |
| fx_shock | 소매 +81원, 환율 +10% | ok | 13.07 | 16.37 | ok |
| tax_policy | 소매 +53원, 인하율 10→0% | ok | 14.08 | 15.08 | ok |
| supply_disruption | 소매 +120원 (입력엔 신호 없음) | ok | 38.83 | 39.11 | ok |

**입력 급변 ≠ 드리프트.** 유가 +35%로 모델 오차가 7원 → 35원으로 커졌지만, 같은 기간 naive는 52원으로 더 크게 틀렸다.
고정 임계값(예: 10원)이었다면 4개 시나리오 모두 재학습했을 것이다. 공급 차질은 입력에 신호가 없어 모델·naive가 비슷하게 틀린다(38.83 vs 39.11, 경계).

## 결과 2: 모델이 학습한 관계가 낡은 경우 (STUB_BIAS=40, 개념 드리프트 흉내)

| 시나리오 | 판정 | 1주차 RMSE | naive RMSE | 조치 | 승격 | 서빙 교체 |
|---|---|---|---|---|---|---|
| normal | **drift** | 42.47 | 10.64 | promoted | 4 | champion:4 |
| oil_shock | ok | 42.64 | 52.25 | ok | - | - |
| fx_shock | **drift** | 40.02 | 16.37 | promoted | 5 | champion:5 |
| tax_policy | **drift** | 38.98 | 15.08 | promoted | 6 | champion:6 |
| supply_disruption | ok | 27.77 | 39.11 | ok | - | - |

평온한 시기(normal)에도 drift가 잡힌다. 반대로 유가 급등기에는 naive가 더 크게 틀려 낡은 모델의 오차가 가려진다(oil_shock ok).
naive 비교 방식의 한계이며, 급변기 직후 평온기에 다시 판정되므로 놓친 드리프트는 곧 잡힌다.

### aiops.log (결과 2의 normal)

```
[WARNING] [WARN] drift detected (week1_rmse=42.47 > naive_rmse=10.64, n=28, period=2026-04-30~2026-05-27, model=champion:3)
[INFO] [INFO] retrain triggered (rows=627, data=2024-06-12~2026-02-28)
[INFO] [OK] new_rmse=[18.00, 30.00, 41.00, 52.00] naive=[35.00, 60.00, 80.00, 90.00] - champion promoted: DieselPricePredictor v4
[WARNING] [ALERT] [AIOps INFO] promoted | ... | 원인: 1주차 RMSE 42.47 > 같은 기간 naive RMSE 10.64 (모델 champion:3, 기간 2026-04-30~2026-05-27) | 조치: 새 모델 v4 승격
```

### 운영자 알림 수신 (mock_webhook, `logs/alerts_received.jsonl`)

```
[MOCK] 알림 수신
[AIOps INFO] promoted | 2026-10-01T07:23:19+00:00
원인: 1주차 RMSE 42.47 > 같은 기간 naive RMSE 10.64 (모델 champion:3, 기간 2026-04-30~2026-05-27)
조치: 새 모델 v4 승격 (서빙 교체는 batch_test reload 결과 확인)
```
본문 JSON은 Slack Incoming Webhook 형식(`text`) + 구조화 필드(`alert`: time·level·status·cause·week1_rmse·threshold_naive_rmse·model_version·period·new_version·new_rmse·action).

## 결과 3: 전송 실패와 중복 (mock_webhook `--fail`, 같은 배치 2회)

```
[INFO] [OK] ... champion promoted: DieselPricePredictor v4
[WARNING] [ALERT] [AIOps INFO] promoted | ...          ← 로그 채널은 정상 기록
[ERROR] [ERROR] alert send failed via webhook: <HTTPError 500: 'Internal Server Error'>
[WARNING] [WARN] drift detected (... period=2030-04-30~2030-05-27 ...)   ← 같은 배치 재전송
[INFO] [INFO] retrain skipped - already attempted for period=2030-04-30~2030-05-27
```
- 웹훅이 500이어도 배치 응답은 200, 재학습 결과는 그대로 반환된다.
- 같은 판정으로는 재학습·알림 모두 반복되지 않는다 (재학습: 이전 결과 재사용 + `reasons`에 "같은 판정은 재학습하지 않음", 알림: 같은 상태·버전·기간 3600초 억제 - `tests/test_notifier.py`).

## 결과 4: 실모델·실측 데이터 재현 (MLflow champion, 실제 fine_tune)

스텁이 아니라 실제 경로로 전체 루프를 돌렸습니다. `diesel_registry.train_and_register` → MLflow `DieselPricePredictor` v1(champion) →
`MODEL_SOURCE=mlflow` 서빙 → 실측 CSV 구간 재전송(`--replay-csv`) → 실제 `fine_tune(최근 627행)` → 게이트 → 알림(mock webhook).

```bash
python -m serving_app.diesel_registry --csv data/processed/diesel_features_2008_spliced.csv   # v1 champion
MODEL_SOURCE=mlflow LOADING_MODE=eager AIOPS_ALERT_WEBHOOK_URL=http://127.0.0.1:9019 uvicorn serving_app.main:app --port 8011
python scripts/simulate_drift.py --url http://127.0.0.1:8011/predict/batch-test \
  --replay-csv data/processed/diesel_features_2008_spliced.csv --replay-end 2026-02-25 2026-07-28 2026-07-28
```

모델 v1: 학습 타깃 ~2025-08-05, 검증 2025-09 ~ 2026-09 1주차 RMSE 27.6원 (naive 28.8원) - 게이트 통과.

| 재현 구간 (기준일) | 판정 | 1주차 RMSE | naive RMSE | 조치 |
|---|---|---|---|---|
| 2026-01-01 ~ 01-28 | ok | 1.77 | 7.13 | - |
| 2026-06-03 ~ 06-30 | **drift** | 26.95 | 25.49 | 실제 fine_tune → **gate_failed**, champion:1 유지 |
| 같은 구간 재전송 | drift | 26.95 | 25.49 | 재학습 없이 이전 결과 재사용 |

```
[WARN] drift detected (week1_rmse=26.95 > naive_rmse=25.49, n=28, period=2026-06-03~2026-06-30, model=champion:1)
[INFO] retrain triggered (rows=627, data=2025-01-12~2026-09-30)
[GATE_FAILED] new_rmse=[16.09, 14.91, 25.70, 41.00] - 1주차 RMSE 16.09 >= naive 15.50; 1~4주 평균 RMSE 24.42 > Production 24.16 (Production 유지)
[ALERT] [AIOps WARN] gate_failed | ... | 조치: 기존 Production 유지: 1주차 RMSE 16.09 >= naive 15.50; ...
[WARN] drift detected (... 같은 판정 ...)
[INFO] retrain skipped - already attempted for period=2026-06-03~2026-06-30
```

추가 재현 (오래된 모델 v1: 학습 타깃 ~2024-01-04, 별도 MLflow DB)
- 2025-05 ~ 2026-01 네 구간: 모두 ok. 2025-05는 모델 2.57 vs naive 1.97로 naive보다 나쁘지만 하한 10원 이하라 ok (하한 효과 확인).
- 2026-03-10 ~ 04-06 (최고가격제 직후): drift (45.21 vs 40.60) → 실제 fine_tune(데이터 ~2026-05-04) → gate_failed
  (1주차 52.52 > 50, 2·3주차 naive보다 나쁨) → Production 유지·알림.

### 실제 실행에서 발견한 것

1. **MLflow 버전 번호 타입 버그 (main `diesel_registry.py`)** - MLflow 3.16은 `ModelVersion.version`을 `int`로 돌려준다.
   `fine_tune()`의 `FineTuneResult(production_before=1)`이 문자열 검증에 걸려 **Production이 있으면 재학습이 항상 실패**했다.
   AIOps는 이를 `retrain_failed`로 받아 Production 유지·운영자 알림까지 정상 처리했다 (실패 복구 경로 실증).
   `production_version()`과 승격 `version`을 `str()`로 바꾸면 해결된다 - 별도 수정 PR로 제안.
2. **fine-tuning 효과가 거의 없음 (#11 참고)** - 세 번 모두 재학습 모델의 1~4주 평균 RMSE가 Production과 0.01~0.3원 차이
   (24.42 vs 24.16, 11.43 vs 11.42, 115.96 vs 115.92). lr 1e-4 · 10 epoch warm start로는 가중치가 거의 움직이지 않아
   "평균 ≤ Production" 조건에서 근소하게 탈락한다. 승격이 사실상 일어나지 않으므로 학습률·epoch 조정 검토가 필요하다.
3. **2026 충격기에는 재학습해도 naive를 못 이긴다** - `docs/evidence/06`과 같은 결론. 이 경우 게이트가 기존 모델을 지키는 것이 맞는 동작이다.

## 한계와 다음 단계

- 결과 1~3은 스텁 모델·가짜 fine_tune 결과다. 실모델 경로는 결과 4, 판정 기준 근거는 `aiops_15_판정기준_백테스트.md`.
- 실모델로는 승격·서빙 교체까지 간 사례가 없다 (위 2·3). 교체 경로는 스텁 E2E·HTTP 통합 테스트와 #34의 실제 pyfunc 교체 검증으로 확인했다.
- batch_test가 내부 예측 윈도우에 서빙 모델 버전을 함께 기록하므로, 승격·교체 후 새 모델(champion:4)은 이전 모델 오차와 섞이지 않고 따로 판정된다.
