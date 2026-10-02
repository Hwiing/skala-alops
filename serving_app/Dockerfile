# 경유 주간 예측 서빙 컨테이너 이미지 (단일 컨테이너)
# 빌드/실행은 반드시 프로젝트 루트에서: docker build -f serving_app/Dockerfile -t diesel-serving .
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

# 이미지에는 모델·데이터를 넣지 않는다. v2 모델은 실제 경유 CSV(data/processed, .dockerignore 제외)로
# 컨테이너 안에서 학습·등록하고, 산출물은 볼륨에 남긴다 (docs/contracts.md "Docker 실행·영속화").
RUN mkdir -p data/uploads data/processed serving_app/models runtime logs mlruns

EXPOSE 8000

ENV MODEL_SOURCE=mlflow
ENV LOADING_MODE=lazy

CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
