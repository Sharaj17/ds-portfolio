# ---------- Base image ----------
# A small official Python 3.12 image (Debian slim). Good balance of size + compatibility.
FROM python:3.12-slim

# ---------- Python runtime behavior ----------
# Don’t write .pyc files to disk; send logs straight to stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------- Working directory inside the container ----------
# All next commands run relative to /app
WORKDIR /app

# ---------- OS packages (minimal for now) ----------
# build-essential is handy for packages that need compiling (kept small & removed cache)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install Python dependencies ----------
# Copy only requirements first for better layer caching.
COPY requirements-api.txt /app/requirements-api.txt

# Upgrade pip tooling, then install exactly what's in requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements-api.txt

# ---------- Copy application code ----------
# Copy only what we need: source code and (optionally) model file(s)
COPY src /app/src
# Option A (self-contained image): copy your model into the image
# If your model lives under notebooks/models, copy that folder in
COPY notebooks/models /app/notebooks/models

# ---------- Environment defaults ----------
# By default we’ll load the local joblib model you copied into the image
ENV USE_MLFLOW=0
ENV MODEL_PATH=/app/notebooks/models/rf_iris.pkl

# ---------- Expose port & default command ----------
EXPOSE 8000

# Use uvicorn to serve FastAPI app
CMD ["uvicorn", "src.models.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
