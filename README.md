# Data Science Portfolio

This repository showcases end-to-end **data science projects** across four evergreen niches:
1. Fraud Detection (Finance/Banking)
2. Cybersecurity Threat Detection
3. Supply Chain & Logistics Optimization
4. Grid/Utility Asset Reliability (Energy)

---

## 📅 Programme Timeline
**4 Weeks | Beginner-friendly | 1h weekdays, 5–8h weekends**

- Week 1: Foundations, Streaming, Anomaly Detection
- Week 2: Graphs, Cybersecurity ML
- Week 3: Forecasting & Optimization
- Week 4: Grid Reliability & Full MLOps

---

## 📂 Repo Structure
- `docs/` → Setup, weekly logs, references, architecture diagrams
- `data/` → Small datasets (or references to external data)
- `notebooks/` → Exploratory notebooks, split by niche
- `src/` → Reusable code, models, pipelines
- `experiments/` → MLflow logs & experiment notes
- `deployment/` → Docker, Kubernetes configs
- `README.md` → High-level overview, progress log

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

