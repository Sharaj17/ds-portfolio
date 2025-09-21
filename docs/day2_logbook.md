# Day 2 Logbook — API Serving and Dockerization

## Goals
- Move from Jupyter-only experimentation to a **production-style API service**.
- Add input validation, automated testing, and containerization.
- Make the project reproducible and shareable.

---

## 1. FastAPI Service
We created `src/models/predict_api.py`:
- **Endpoints**
  - `GET /health` → `{"status":"ok"}` for monitoring.
  - `POST /predict` → predicts class ID (and later, labels + probs).
  - `POST /predict_batch` → batch predictions.

- **Key details**
  - Uses **Pydantic models** for input validation.
  - Loads the model once at startup (`rf_iris.pkl`).
  - Swagger UI auto-available at `/docs`.

---

## 2. Automated Tests
We created `tests/test_api.py` with pytest:
- `/health` returns 200 and status ok.
- `/predict` with valid features returns 200 and a class id.
- `/predict` with invalid features returns 422.

**Fixes along the way:**
- Needed to add `PYTHONPATH` or `conftest.py` so Python could find `src/`.
- Made sure `MODEL_PATH` env var points to `notebooks/models/rf_iris.pkl`.

---

## 3. Dockerization
We created `deployment/Dockerfile`:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements-api.txt

COPY src /app/src
COPY notebooks/models /app/notebooks/models

ENV USE_MLFLOW=0
ENV MODEL_PATH=/app/notebooks/models/rf_iris.pkl

EXPOSE 8000

CMD ["uvicorn", "src.models.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
