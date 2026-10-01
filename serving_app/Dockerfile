# Day2: 휘발유 서빙 앱 컨테이너 이미지 (단일 컨테이너)
# 빌드/실행은 반드시 프로젝트 루트에서: docker build -f serving_app/Dockerfile -t gasoline-serving:v1 .
# 참고: tensorflow가 들어가서 이미지가 (scikit-learn만 쓸 때보다) 훨씬 큽니다
#       (1GB대) - 빌드에 몇 분 걸릴 수 있습니다. (GPU 없는 컨테이너이지만 CPU로 정상 동작합니다)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# fastapi, uvicorn, mlflow, tensorflow, python-multipart(파일 업로드) 포함

COPY serving_app/ serving_app/
COPY data/ data/
COPY scripts/ scripts/
ENV MLFLOW_TRACKING_URI=sqlite:////app/runtime/mlflow.db

# 원본 seed → baseline → MLflow 등록 흐름은 유지하되 담당자 TODO 완료 후 선택 실행.
# 합성 데이터로 자동 학습/Production 승격하지 않는 뼈대 기본값.
ARG TRAIN_ON_BUILD=0
RUN mkdir -p data/uploads serving_app/models runtime logs mlruns \
 && if [ "$TRAIN_ON_BUILD" = "1" ]; then \
      cp data/sample_diesel_prices.csv data/uploads/build_seed.csv && \
      python scripts/train_baseline_v1.py && \
      python serving_app/train_and_register.py; \
    fi

EXPOSE 8000

ENV MODEL_SOURCE=local
ENV LOADING_MODE=lazy

CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
