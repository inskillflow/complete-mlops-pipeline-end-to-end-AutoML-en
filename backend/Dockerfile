# Backend image: FastAPI + H2O + MLflow (also reused for the mlflow server and trainer)
FROM python:3.11-slim

WORKDIR /app

# H2O needs a Java runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# 8000 = FastAPI, 5000 = MLflow server (when this image runs the mlflow command)
EXPOSE 8000 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
