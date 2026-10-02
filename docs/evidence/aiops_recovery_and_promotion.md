# AIOps 복구·최신 데이터·운영자 수신함·자동 승격 검증

관련: #15, #16, #17, #14. 2026-10-01 검증. 최신 main `728371b`의 AIOps 후속 수정이다.

## 변경과 완료 기준

- 동일 판정 경합: 잠금 획득 뒤 결과·데이터 지문을 확인하고 결과 기록까지 잠금 유지. 동일 입력의 동시 요청은 학습 1회.
- 복구: `promoted`·`gate_failed`만 캐시. 데이터 부족·학습 예외·Production 없음은 쿨다운 뒤 같은 판정도 재시도.
- 데이터: 설정 CSV·모든 업로드 중 공통 검증과 최소 627행을 통과한 후보를 비교. 가장 최근 마지막 날짜 선택, 같은 날짜는 설정 CSV 우선. 마지막 627행의 정규화 JSON SHA-256을 로그·중복 키에 기록. 데이터가 달라지면 쿨다운 뒤 다시 학습.
- 업로드: 저장 완료까지 `.pending` 확장자로 두고 원자적으로 공개. 검증·저장·지연 정답·재학습은 작업 스레드로 이동. 다른 health 요청은 재학습 중에도 처리됨.
- 운영자 채널: 기본 `logs/alerts.jsonl` 영속 수신함과 기존 로그. 선택적 HTTP 웹훅. Registry 승격과 실제 서빙 교체 결과는 별도 알림. 외부 채널 실패가 수신함·예측 응답을 막지 않음.

## 재현

전체 의존성(TensorFlow·MLflow)을 설치한 저장소 루트에서:

```bash
python scripts/verify_aiops_loop.py --out /tmp/skala-aiops-loop.json
```

스크립트는 임시 폴더에 정책표 사본·합성 CSV·SQLite·모델·로그를 만들고, 임의 로컬 포트의 실제 Uvicorn과 모의 HTTP 웹훅을 실행한다. 종료 시 정리하며 기존 Registry·CSV는 사용하거나 수정하지 않는다. 전체 응답과 로그는 지정한 JSON으로 남긴다.

이번 검증은 Python 3.11 / TensorFlow 2.21.0 / MLflow 3.16.0의 기존 Docker 이미지에서 수정 소스를 읽기 전용 mount해 실행했다.

```bash
docker run --rm \
  --mount type=bind,src="$PWD",dst=/source,readonly \
  --mount type=bind,src=/tmp,dst=/evidence \
  --workdir /source --entrypoint python serving_app-serving-app:latest \
  scripts/verify_aiops_loop.py --out /evidence/skala-aiops-loop.json
```

## 실제 실행 결과

합성 700행, 가격 하루 +3원, 정책 변경이 없는 2014~2015 구간. 초기 v1은 음수 가격 변화 출력을 가지도록 준비한 stale LSTM fixture를 Registry에 직접 등록했다. 신규 v2는 실제 `fine_tune()`(627행, 365개 학습 기준일, 독립 90개 검증, 10 epochs, lr=1e-4)과 변경하지 않은 배포 게이트를 통과해 등록됐다. fine_tune·gate·reload 대역은 사용하지 않았다.

| 단계 | 확인 결과 |
|---|---|
| 최초 `/predict` | `champion:1` |
| 700행 CSV 업로드 | 정상 저장, 설정 CSV보다 하루 최신인 업로드 선택 |
| 175행 배치 | 기준일 28개, drift RMSE 12.02 > naive 12.00, 하한 10원/L도 초과 |
| 학습 입력 | 2014-03-16~2015-12-02, 627행 |
| 입력 지문 | `4ddb402bf9d856cbfa57f3faf94b481ae9785ce15105690be0182d6ed32c11c5` |
| 신규 주별 RMSE | `[11.9154, 32.9153, 53.9153, 74.9153]` |
| naive 주별 RMSE | `[12.0, 33.0, 54.0, 75.0]` |
| 기존 Production 주별 RMSE | `[12.02, 33.02, 54.02, 75.02]` |
| 게이트 | 1주 ≤50, 4주 각각 naive 우위, Production 평균보다 우수 → 통과 |
| 실제 fine-tune run | `985bd92c03b1465bad08cb4c9b333d31` (격리 SQLite의 실행 식별자) |
| 배치 결과 | `promoted=true`, `version="2"`, `reload.reloaded=true` |
| 후속 `/predict`·`/health` | 모두 `champion:2` |
| 같은 배치 재전송 | `ok`, drift RMSE 11.92 < naive 12.00, 추가 학습 없음 |
| 실제 운영자 수신함 조회 | `/logs/alerts.jsonl`: `promoted`, `reloaded` 각각 1건 |
| 모의 HTTP 웹훅 | 실제 POST로 같은 2건 수신 |

원본 응답·로그: [합성 통합 검증 JSON](aiops_synthetic_loop_result.json). 최종 코드 재실행에서도 승격·교체·알림 수신을 확인했다.

## 운영자 사용

기본 채널은 외부 계정 없이 준비되며 기존 logs 볼륨에 저장된다.

```bash
curl http://localhost:8000/logs/alerts.jsonl
```

외부 Slack 등은 운영자가 해당 채널의 Incoming Webhook을 만든 뒤 `AIOPS_ALERT_WEBHOOK_URL`을 서버/Compose 환경에 넣고 재기동한다. 비밀 URL은 Git·증빙에 넣지 않는다. 이번에 외부 Slack 워크스페이스·채널을 만들거나 실제 Slack으로 전송하지는 않았다. 기본 영속 수신함은 실제 검증했고 외부 전송 경계는 모의 HTTP 수신으로 검증했다.

## 범위와 한계

- 이 수치는 **합성 fixture의 기능 연결 증빙**이다. 약한 초기 모델을 의도적으로 구성했으며 실측 성능 개선·실측 재학습 승격의 증거가 아니다. 실측 fine-tuning gate_failed 증빙은 기존 `10_실측데이터_MLflow_fine_tuning_검증.md`와 구분한다.
- 과거 배치를 보내도 선택된 데이터의 최신 627행으로 학습한다. 과거 시점의 재학습 성능을 재현하려면 당시 cutoff에 맞는 데이터·모델·정책표를 별도로 준비해야 한다.
- 예측 윈도우·재학습 잠금·캐시는 단일 서버 프로세스 메모리다. 서버 재시작 시 판정 이력이 초기화되고 다중 worker의 분산 잠금을 제공하지 않는다. 운영자 수신함 파일은 기존 logs 볼륨에 영속된다.
- 업로드 화면에서 `drift_check`를 표시하는 UI 개선과 실측 인계본의 세율·해시 재생성은 각 담당의 후속이다.
