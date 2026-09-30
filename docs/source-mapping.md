# 출처와 변경 범위

기반: 사용자가 보유한 개인 실습 `project/`를 **직접 복사**했습니다.
참고 문서: `SKALA_모델서빙_AIOps_실습가이드_v2.pdf`, `SKALA_AIOps_context.md`.
첨부 문서는 요구사항 참고 자료로 사용하며 문서의 개인 실습 제출·Slack 전송 안내는 이 저장소 작업 범위에 포함하지 않습니다.
원본 자료와 PDF 자체는 저장소에 게시하지 않습니다.

| 원본 | 재사용 및 변경 |
|---|---|
| `main.py`, `routers/health.py` | 앱 구성, 로그 설정, 정적 대시보드, Lazy/Eager 흐름 유지 |
| `routers/data.py`, `data/storage.py` | 업로드·최신 파일·상태 조회 유지, 휘발유 CSV 계약과 값 검증 반영 |
| `routers/logs.py`, `static/` | 로그 조회와 대시보드 유지, 도메인 문구·배치 요청·단위 변경 |
| `data/features.py` | min-max scaler·시퀀스·시간순 분할 구조 유지, 4피처로 확장 |
| `schemas.py`, `model_loader.py` | 입력/출력 변경, local/MLflow 로더 구조와 캐시 유지 |
| `lstm_model.py` | 기존 LSTM 아키텍처 유지, 입력 피처 수만 4개로 변경 |
| `train_baseline_v1.py` | 기존 학습 유지, scaler를 학습 구간에만 fit |
| `train_and_register.py` | Tracking·Registry·fine-tuning 유지, 10원 및 naive 비교 게이트·SQLite 명시 |
| `monitoring/`, `simulate_drift.py` | 감지→재학습 구조와 실습 TODO 유지, 도메인 정책 연결 지점 명시 |
| Dockerfile / compose | 단일 컨테이너 및 seed→학습→등록 흐름 유지; 뼈대 기본은 학습 없이 lazy 기동 |

원본 핵심 TODO 5개: `_load_from_mlflow`, `batch_test`, `compute_rmse`, `check_and_trigger`, `send_batch`.
구현되지 않은 배치/재학습이 성공처럼 응답하지 않도록 명시적 미구현 오류를 사용합니다.
새로 추가한 것은 팀 계약, 합성 CSV, 요청 예제, Makefile, CI, 기본 회귀 테스트입니다.
기존 주식 예제의 학습 시간·성능은 휘발유 성능 근거가 아니며 모두 새로 측정해야 합니다.
