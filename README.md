# Data Science Portfolio Framework

This repository contains the **end-to-end framework of a production-grade ML system**.  
We use the classic **Iris classification problem** as a teaching example, but the same setup applies to **real-world projects** (fraud detection, supply chain, energy reliability, etc.).

The goal of this repo is to demonstrate **industry-standard MLOps practices**:
- Experiment tracking (MLflow)
- API service (FastAPI + Pydantic)
- Automated testing (pytest + GitHub Actions CI)
- Containerization (Docker)
- Observability (logging, error handling)
- Deployment (Kubernetes + GHCR)

---

## 📂 Repo Structure
- `docs/` → Setup, logs, architecture diagrams
- `data/` → Datasets (or references)
- `notebooks/` → Exploration & training
- `src/` → Reusable API + model code
- `tests/` → Automated tests
- `experiments/` → MLflow logs
- `deployment/` → Dockerfile, Kubernetes manifests
- `README.md` → This overview + progress log

---

## ✅ Progress Log
- Day 1: Repo setup & baseline MLflow experiment
- Day 2: API service + Dockerization
- Day 3: Continuous Integration with GitHub Actions
- Day 4–5: Logging & Error Handling
- Day 6: Kubernetes Deployment with GHCR

---

## ✅ Progress Log
- Day 1: Repo setup & baseline MLflow experiment
### Day 1 — Baseline Experiment
- RandomForest on Iris with stratified 60/20/20 split.
- Logged hyperparameters, validation/test metrics, confusion matrix, and model to MLflow.
- Compared different settings (`n_estimators=200 vs 50`, `max_depth=None vs 4`) to see bias/variance tradeoffs.
- Learned how to use MLflow UI to track and compare experiments.
### Day 2 — API & Dockerization
- Built a FastAPI service (`src/models/predict_api.py`) exposing:
  - `GET /health` → service status.
  - `POST /predict` → single prediction.
  - `POST /predict_batch` → multiple predictions.
- Validated inputs with **Pydantic models**; Swagger UI auto-generated at `/docs`.
- Added **pytest tests** (`tests/test_api.py`) to check endpoints with valid/invalid inputs.
- Solved import path issues with `PYTHONPATH` / `conftest.py`.
- Dockerized the service:
  - Created `deployment/Dockerfile` + `.dockerignore`.
  - Split dependencies (`requirements-api.txt` for container vs `requirements.txt` for dev).
  - Built and ran container: `docker run -p 8000:8000 ds-portfolio-api`.
- Verified service works in container (Swagger UI shows `/health` → `{"status": "ok"}`).
- Documented challenges & fixes (Windows-only packages, virtualization, imports).
- Day 3: Continuous Integration (CI) with GitHub Actions

### Day 3 — Continuous Integration
- Added `.github/workflows/ci.yml` to automatically run tests on every push/PR.  
- Configured GitHub Actions with Python 3.12, installed dependencies, and executed pytest suite.  
- Fixed cross-platform issues: added `pytest` + `httpx` to requirements, guarded `pywin32` with platform markers.  
- Made `src` a package (`__init__.py` files) and added `tests/conftest.py` for clean imports.  
- Committed model artifact (`rf_iris.pkl`) so CI can load it in a fresh VM.  
- All tests now run in CI — repo shows a ✅ green badge when passing.  
- Learned how to interpret common CI errors (`exit 127`, `exit 5`, import errors) and fix them.  

### Day 4 and 5 — Logging & Error Handling
- Added structured logging with Python’s `logging` module.
- Implemented request logging middleware:
  - Generates unique `request_id` per request.
  - Logs start, end, duration, and status code.
  - Propagates `X-Request-ID` header in responses.
- Built global exception handler:
  - Catches all unhandled exceptions.
  - Logs full stacktrace.
  - Returns clean JSON error with `request_id`.
- Added pytest tests:
  - Validated logging does not break API.
  - Confirmed JSON error response for failing route (`/_boom`).
  - Used `raise_server_exceptions=False` for testing error paths.
- Silenced noisy deprecation warnings with `pytest.ini`.
- Verified all tests pass locally and in CI.
- Updated documentation (`MASTER_KEY_EXPANDED.md`) with full detailed knowledge base.

### Day 6 — Kubernetes Deployment with GHCR
- Pushed Docker image to **GHCR** (`ghcr.io/sharaj17/ds-portfolio-framework:latest`).
- Created **Kubernetes Deployment** with 2 replicas, rolling updates, probes, and resource limits.
- Added **Service** (NodePort + port-forward) to expose the API.
- Verified:
  - `/health` → `{"status":"ok"}`
  - `/predict` → returned predictions on sample inputs.
- Practiced:
  - `kubectl logs`, `kubectl scale`, `kubectl rollout status`, `kubectl rollout undo`.

**Key Learning:**  
The same system works for any ML project — Iris here is a toy example.  
Next step: extend this framework with **real-world use cases**.

---

## 🚀 Next Steps
- Add fraud detection, supply chain optimization, and energy reliability projects into this framework.
- Each project will have:
  - Dataset & training notebook
  - MLflow-tracked experiments
  - FastAPI endpoints
  - Tests & CI
  - Containerized & deployed on Kubernetes