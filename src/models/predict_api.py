from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist
from typing import List, Optional
import os
import joblib
import pandas as pd

# Optional MLflow loading
USE_MLFLOW = os.getenv("USE_MLFLOW", "0") == "1"

app = FastAPI(
    title="Iris RF Baseline API",
    description="Simple FastAPI service that wraps a RandomForest Iris model.",
    version="0.1.0",
)

# ----- Input schema -----
class IrisInput(BaseModel):
    # Four numeric features, matching scikit-learn Iris order:
    # sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
    #features: List[float] = Field(..., description="Exactly four numeric features in the Iris order.", min_items=4, max_items=4)
    features: conlist(float, min_length=4, max_length=4) = Field(..., description="Exactly four numeric features in the Iris order.")
class BatchIrisInput(BaseModel):
    batch: List[IrisInput]

class PredictionOutput(BaseModel):
    prediction: int
    # You can add probabilities later if you want

# ----- Model load -----
model = None

def load_model():
    global model
    if USE_MLFLOW:
        try:
            import mlflow.pyfunc
            model_uri = os.getenv("MODEL_URI")
            if not model_uri:
                raise RuntimeError("MODEL_URI env var required when USE_MLFLOW=1")
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"[LOAD] Loaded MLflow model from {model_uri}")
        except Exception as e:
            raise RuntimeError(f"Failed to load MLflow model: {e}")
    else:
        # Default: local pickle
        path = os.getenv("MODEL_PATH", "notebooks/models/rf_iris.pkl")
        try:
            model = joblib.load(path)
            print(f"[LOAD] Loaded joblib model from {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load joblib model from {path}: {e}")

@app.on_event("startup")
def _startup():
    load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict(item: IrisInput):
    """
    Predict the Iris class.
    - Input: IrisInput with 'features' of length 4
    - Output: class index (0,1,2)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = pd.DataFrame([item.features], columns=[
        "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"
    ])

    try:
        # If MLflow pyfunc model → .predict returns numpy array or DataFrame
        # If scikit-learn model → same interface
        y_pred = model.predict(X)
        pred = int(y_pred[0])
        return {"prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.post("/predict_batch")
def predict_batch(items: BatchIrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = pd.DataFrame([it.features for it in items.batch], columns=[
        "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"
    ])

    try:
        y_pred = model.predict(X)
        return {"predictions": [int(v) for v in y_pred]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

__all__ = ["app"]
