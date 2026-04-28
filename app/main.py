import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.model_loader import load_production_model
from app.feature_client import get_feature_store, get_online_features, MODEL_FEATURE_NAMES
from app.metrics import (
    prediction_requests_total,
    prediction_duration_seconds,
    prediction_errors_total,
    model_version_info,
    feast_lookup_total,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = None
MODEL_VERSION = "unknown"
FEATURE_STORE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, MODEL_VERSION, FEATURE_STORE

    logger.info("Starting up — loading model and feature store...")
    MODEL, MODEL_VERSION = load_production_model()
    FEATURE_STORE = get_feature_store()
    model_version_info.labels(version=MODEL_VERSION).set(1)
    logger.info(f"Ready! Model version: {MODEL_VERSION}")
    yield
    logger.info("Shutting down...")


app = FastAPI(title="ModelServe — Fraud Detection API", lifespan=lifespan)


class PredictRequest(BaseModel):
    entity_id: int


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    timestamp: str


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.time()

    try:
        try:
            features = get_online_features(FEATURE_STORE, request.entity_id)
            feast_lookup_total.labels(result="hit").inc()
        except Exception as e:
            feast_lookup_total.labels(result="miss").inc()
            prediction_errors_total.inc()
            prediction_requests_total.labels(status="error").inc()
            raise HTTPException(
                status_code=503,
                detail={"error": "Feature lookup failed", "message": str(e)},
            )

        feature_vector = [features.get(col, 0.0) for col in MODEL_FEATURE_NAMES]
        X = np.array(feature_vector).reshape(1, -1)
        prediction = int(MODEL.predict(X)[0])
        probability = float(MODEL.predict_proba(X)[0][1])

        duration = time.time() - start
        prediction_duration_seconds.observe(duration)
        prediction_requests_total.labels(status="success").inc()

        return PredictResponse(
            prediction=prediction,
            probability=probability,
            model_version=MODEL_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        prediction_errors_total.inc()
        prediction_requests_total.labels(status="error").inc()
        raise HTTPException(
            status_code=500,
            detail={"error": "Prediction failed", "message": str(e)},
        )


@app.get("/predict/{entity_id}")
def predict_explain(entity_id: int, explain: bool = False):
    """GET version — returns prediction + optional feature values when explain=true."""
    try:
        features = get_online_features(FEATURE_STORE, entity_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    feature_vector = [features.get(col, 0.0) for col in MODEL_FEATURE_NAMES]
    X = np.array(feature_vector).reshape(1, -1)
    prediction = int(MODEL.predict(X)[0])
    probability = float(MODEL.predict_proba(X)[0][1])

    response = {
        "prediction": prediction,
        "probability": probability,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if explain:
        response["features_used"] = features

    return response


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
