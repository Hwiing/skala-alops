# 공통 데이터·API·운영 계약

## 데이터

정규화 CSV와 입력 피처 순서는 다음과 같습니다. 오피넷 원본 파일은 데이터 담당자가 이 형식으로 변환합니다.

| 컬럼 | 의미 | 검증 |
|---|---|---|
| `date` | 기준일, ISO 날짜 | 하루 간격 오름차순, 중복·누락 불가 |
| `gasoline_price` | 전국 평균 보통휘발유 가격, 원/L | 양수 |
| `crude_oil_price` | 선정 국제유가, USD/barrel | 양수; 유종·시차는 데이터 담당 확정 |
| `usd_krw` | 원/달러 환율, KRW/USD | 양수 |
| `tax_or_supply_feature` | 유류세/공급 지표 | 유한 실수; 정의·단위는 데이터 담당 확정 |

합성 예제의 정책 피처 0은 실제 관측값이 아닙니다. 추가 지표의 출처·라이선스·발표 시각과
휴일 결측 처리 정책을 확정하기 전에는 실측 성능을 주장하지 않습니다. 미래 값으로 보간하지 않습니다.
API sequence는 가장 오래된 날부터 최근 날까지 20개이며 날짜는 생략합니다.
학습·추론 모두 `FEATURE_COLUMNS` 순서로 `(N,20,4)`를 구성하고 마지막 입력 다음날 가격을 예측합니다.
CSV 업로드 최소 41행은 시뮬레이션 21건 확보 기준이며 충분한 학습량을 의미하지 않습니다.

## API

| Method | URL | 요청 / 응답 | 뼈대 상태 |
|---|---|---|---|
| GET | `/` | 원본 기반 대시보드 | 동작 |
| GET | `/health` | `status`, `model_loaded`, `loading_mode` | 프로세스 상태; 모델 readiness는 별도 확인 |
| POST | `/data/upload` | multipart CSV → `filename`, `rows` | 검증·저장, 오류 400 |
| GET | `/data/status` | 데이터 기간·행 수·`min_price`·`max_price` | 데이터 없으면 `exists:false` |
| POST | `/predict` | `sequence` → `predicted_price`, `model_version` | 모델 필요, 입력 422, 미준비 503 |
| POST | `/predict/batch-test` | `rows` 21개 이상 → `predictions`, `drift_check` | TODO, 현재 501 |
| GET | `/logs` | 로그 목록 | 원본 재사용 |
| GET | `/logs/{filename}` | `name`, `content` | 원본 재사용, 없는 파일 404 |

전체 예측 요청 예시는 [`examples/predict.json`](../examples/predict.json).
배치의 `rows`는 DailyPoint 전체 피처를 포함합니다. 원본 `prices` 전용 계약은 사용하지 않습니다.
배치에서 `rows[i:i+20]`으로 예측하고 `rows[i+20].gasoline_price`를 실제값으로 사용합니다.
응답 `model_version`은 로컬 `v1-local`; MLflow 구현 후 `production:<실제 등록 버전>`으로 식별합니다.

## 모델·운영

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
