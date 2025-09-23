from urllib import request, response
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist
from typing import List, Optional
import os
import joblib
import pandas as pd
import time
import uuid
import logging
from fastapi import Request
from fastapi.responses import JSONResponse

from src.config.logging_config import init_logging

logger = init_logging()
CLASS_LABELS = ["setosa", "versicolor", "virginica"]

# Optional MLflow loading
USE_MLFLOW = os.getenv("USE_MLFLOW", "0") == "1"

# Iris class names in the same order the model uses (0,1,2)
CLASS_LABELS = ["setosa", "versicolor", "virginica"]


app = FastAPI(
    title="Iris RF Baseline API",
    description="Simple FastAPI service that wraps a RandomForest Iris model.",
    version="0.1.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    logger.info(f"req_id={request_id} start {request.method} {request.url.path}")
    request.state.request_id = request_id  # type: ignore[attr-defined]

    response = None
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        # Let the global exception handler format the response; just log here
        logger.exception(f"req_id={request_id} unhandled_exception_in_middleware: {exc}")
        raise
    finally:
        dur_ms = (time.perf_counter() - start) * 1000.0
        status = getattr(response, "status_code", "ERR")
        logger.info(
            f"req_id={request_id} done {request.method} {request.url.path} "
            f"status={status} dur_ms={dur_ms:.1f}"
        )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "NA")
    logger.exception(f"req_id={request_id} unhandled_exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "request_id": request_id,
            "detail": "An unexpected error occurred.",
        },
        headers={"X-Request-ID": request_id},
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
    class_label: str                         # "setosa" | "versicolor" | "virginica"
    probs: Optional[List[float]] = None      # [p_setosa, p_versicolor, p_virginica]

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
def predict(item: IrisInput, request: Request):
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
        response = {"prediction": pred, "class_label": CLASS_LABELS[pred]}

        # Add probabilities if the model supports it (RandomForest does)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0].tolist()   # e.g., [0.97, 0.03, 0.00]
            response["probs"] = proba
        # inside your /predict handler, after computing `pred` and possibly `proba`
        #req_id = getattr(request.state, "request_id", "NA")  # add `request: Request` param to the endpoint
        req_id = getattr(getattr(request, "state", object()), "request_id", "NA")
        logger.info(f"req_id={req_id} predict pred={pred} label={CLASS_LABELS[pred]}")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.post("/predict_batch")
def predict_batch(items: BatchIrisInput, request: Request):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = pd.DataFrame([it.features for it in items.batch], columns=[
        "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"
    ])

    try:
        y_pred = model.predict(X)
        preds = [int(v) for v in y_pred]
        resp = {
            "predictions": preds,
            "class_labels": [CLASS_LABELS[p] for p in preds]
        }
        if hasattr(model, "predict_proba"):
            resp["probs"] = model.predict_proba(X).tolist()  # list of lists
        req_id = getattr(getattr(request, "state", object()), "request_id", "NA")
        logger.info(f"req_id={req_id} predict_batch n={len(items.batch)}")
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.get("/_boom")
def boom():
    # raise only when explicitly enabled (e.g., tests)
    if os.getenv("ENABLE_TEST_ROUTES") == "1":
        raise RuntimeError("Boom for test")
    return {"ok": True}


__all__ = ["app"]
