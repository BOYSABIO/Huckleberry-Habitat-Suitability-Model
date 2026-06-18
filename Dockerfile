FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt 

# Application code
COPY src/ src/

# Default for standalone `docker run` when MLflow runs on the host (must be up first).
# Prefer `docker compose up` — compose sets MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000
ENV MLFLOW_MODEL_URI=models:/huckleberry-habitat@production

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]