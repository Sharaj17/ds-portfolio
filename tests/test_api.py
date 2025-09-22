# tests/test_api.py
import os
from pathlib import Path

# --- 1) Set env *before* importing the app ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "notebooks" / "models" / "rf_iris.pkl"

# Helpful assertion so failures are obvious
assert MODEL_PATH.exists(), f"Model file not found at: {MODEL_PATH}"

os.environ["USE_MLFLOW"] = "0"
os.environ["MODEL_PATH"] = str(MODEL_PATH)

# --- 2) Now import the app ---
from fastapi.testclient import TestClient
from src.models.predict_api import app


def test_health():
    # --- 3) Use context manager so startup runs ---
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_predict_valid():
    with TestClient(app) as client:
        payload = {"features": [5.1, 3.5, 1.4, 0.2]}
        r = client.post("/predict", json=payload)

        # If it fails again, print the body for debugging
        if r.status_code != 200:
            print("FAIL BODY:", r.text)

        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1, 2]


def test_predict_invalid_length():
    with TestClient(app) as client:
        payload = {"features": [5.1, 3.5]}  # only 2 features
        r = client.post("/predict", json=payload)
        assert r.status_code == 422
