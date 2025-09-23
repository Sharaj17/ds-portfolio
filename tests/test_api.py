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
        assert r.status_code == 200
        data = r.json()

        # basic checks
        assert data["prediction"] in [0, 1, 2]
        assert data["class_label"] in ["setosa", "versicolor", "virginica"]

        # optional but nice: probabilities sanity
        if "probs" in data:
            probs = data["probs"]
            assert isinstance(probs, list)
            assert len(probs) == 3
            # they are probabilities between 0 and 1 and sum ~ 1
            assert all(0.0 <= p <= 1.0 for p in probs)
            assert abs(sum(probs) - 1.0) < 1e-6

def test_predict_batch_valid():
    with TestClient(app) as client:
        payload = {
            "batch": [
                {"features": [5.1, 3.5, 1.4, 0.2]},
                {"features": [6.2, 3.4, 5.4, 2.3]}
            ]
        }
        r = client.post("/predict_batch", json=payload)
        assert r.status_code == 200
        data = r.json()

        assert "predictions" in data and "class_labels" in data
        assert len(data["predictions"]) == 2
        assert len(data["class_labels"]) == 2
        assert set(data["class_labels"]).issubset({"setosa", "versicolor", "virginica"})

        if "probs" in data:
            assert len(data["probs"]) == 2
            for row in data["probs"]:
                assert len(row) == 3
                assert all(0.0 <= p <= 1.0 for p in row)
                assert abs(sum(row) - 1.0) < 1e-6

def test_predict_invalid_length():
    with TestClient(app) as client:
        payload = {"features": [5.1, 3.5]}  # only 2 features
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

def test_error_handler_returns_json():
    import os
    os.environ["ENABLE_TEST_ROUTES"] = "1"
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/_boom")
        assert r.status_code == 500
        data = r.json()
        assert data["error"] == "Internal Server Error"
        assert "request_id" in data
