# AIOps #16 드리프트 시나리오·운영자 알림 검증

> **합성 데이터 + 스텁 모델 결과입니다. 모델 성능 증빙이 아닙니다.**
> 목적은 감지 → 재학습 → 승격 → 알림 → 서빙 교체 흐름과 판정 기준(1주차 RMSE vs naive)의 동작 확인입니다.
> 실모델 결과는 `data/processed` 경유 데이터와 MLflow champion 모델로 같은 명령을 다시 돌려 교체합니다.

## 조건

- 브랜치: `feat/16-drift-scenarios` + #34(`feat/14-batch-test`) 로컬 병합 (`scripts/simulate_drift.py`는 #16 버전 사용)
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
[WARNING] [WARN] drift detected (week1_rmse=42.47 > naive_rmse=10.64, n=28, period=2026-04-30~2026-05-27, model=unknown)
[INFO] [INFO] retrain triggered (rows=627, data=2024-06-12~2026-02-28)
[INFO] [OK] new_rmse=[18.00, 30.00, 41.00, 52.00] naive=[35.00, 60.00, 80.00, 90.00] - champion promoted: DieselPricePredictor v4
[WARNING] [ALERT] [AIOps INFO] promoted | ... | 원인: 1주차 RMSE 42.47 > 같은 기간 naive RMSE 10.64 (모델 unknown, 기간 2026-04-30~2026-05-27) | 조치: 새 모델 v4 승격
```

### 운영자 알림 수신 (mock_webhook, `logs/alerts_received.jsonl`)

```
[MOCK] 알림 수신
[AIOps INFO] promoted | 2026-10-01T07:23:19+00:00
원인: 1주차 RMSE 42.47 > 같은 기간 naive RMSE 10.64 (모델 unknown, 기간 2031-04-30~2031-05-27)
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
- 같은 판정으로는 재학습·알림 모두 반복되지 않는다 (재학습: `skipped_duplicate`, 알림: 같은 상태·버전·기간 3600초 억제 - `tests/test_notifier.py`).

## 한계와 다음 단계

- 스텁 모델·가짜 fine_tune 결과다. 실모델로 결과 1·2 표를 다시 채워야 판정 기준(28건·naive 비교)의 근거가 된다 (#15 남은 항목).
- #34 짝에 `model_version`이 없어 로그의 `model=unknown`. 추가되면 버전별 판정·중복 키가 정확해진다.
