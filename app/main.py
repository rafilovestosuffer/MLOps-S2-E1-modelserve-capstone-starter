import time
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.logger import configure_logging, get_logger
from app.model_loader import load_production_model
from app.feature_client import get_feature_store, get_online_features, MODEL_FEATURE_NAMES
from app.metrics import (
    prediction_requests_total,
    prediction_duration_seconds,
    prediction_errors_total,
    model_version_info,
    feast_lookup_total,
)

# Initialize logging once at module import
configure_logging()
logger = get_logger(__name__)

MODEL = None
MODEL_VERSION = "unknown"
FEATURE_STORE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, MODEL_VERSION, FEATURE_STORE

    # ──────── STARTUP ────────
    logger.info("ModelServe starting up...")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    logger.info(f"MLflow tracking URI: {mlflow_uri}")

    model_name = os.getenv("MLFLOW_MODEL_NAME", "fraud_detector")
    model_stage = "Production"
    logger.info(f"Loading model: {model_name}@{model_stage}")

    try:
        MODEL, MODEL_VERSION = load_production_model()
        logger.info(f"Model version number: {MODEL_VERSION}")

        FEATURE_STORE = get_feature_store()
        logger.info("Feast feature store initialized successfully")

        model_version_info.labels(version=MODEL_VERSION).set(1)
        logger.info("ModelServe ready to serve predictions")

    except Exception as e:
        logger.exception("Failed to start ModelServe — startup initialization failed")
        raise

    yield

    # ──────── SHUTDOWN ────────
    logger.info("ModelServe shutting down gracefully")


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
    logger.debug(f"Health check — model_version: {MODEL_VERSION}")
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.time()

    logger.info(f"Prediction request received — entity_id={request.entity_id}")

    try:
        try:
            features = get_online_features(FEATURE_STORE, request.entity_id)
            feast_lookup_total.labels(result="hit").inc()
            logger.debug(f"Feature values fetched from Feast: {features}")

        except Exception as e:
            # Check if this is a "not found" error (cache miss)
            if "not found" in str(e).lower() or "missing" in str(e).lower():
                logger.warning(f"Feature cache MISS for entity_id={request.entity_id}")
            else:
                logger.error(f"Feast lookup failed for entity_id={request.entity_id}: {e}")

            feast_lookup_total.labels(result="miss").inc()
            prediction_errors_total.inc()
            prediction_requests_total.labels(status="error").inc()
            raise HTTPException(
                status_code=503,
                detail={"error": "Feature lookup failed", "message": str(e)},
            )

        # Check for null features
        null_fields = [col for col in MODEL_FEATURE_NAMES if features.get(col) is None]
        if null_fields:
            logger.warning(f"Null features detected for entity_id={request.entity_id}, fields: {null_fields}")

        feature_vector = [features.get(col, 0.0) for col in MODEL_FEATURE_NAMES]
        X = np.array(feature_vector).reshape(1, -1)
        prediction = int(MODEL.predict(X)[0])
        probability = float(MODEL.predict_proba(X)[0][1])

        duration = time.time() - start
        latency_ms = int(duration * 1000)
        prediction_duration_seconds.observe(duration)
        prediction_requests_total.labels(status="success").inc()

        logger.info(
            f"Prediction complete — result={prediction}, probability={probability:.4f}, "
            f"version={MODEL_VERSION}, latency={latency_ms}ms"
        )

        return PredictResponse(
            prediction=prediction,
            probability=probability,
            model_version=MODEL_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Exception during prediction for entity_id={request.entity_id}")
        prediction_errors_total.inc()
        prediction_requests_total.labels(status="error").inc()
        raise HTTPException(
            status_code=500,
            detail={"error": "Prediction failed", "message": str(e)},
        )


@app.get("/predict/{entity_id}")
def predict_explain(entity_id: int, explain: bool = False):
    """GET version — returns prediction + optional feature values when explain=true."""
    logger.info(f"Prediction request received — entity_id={entity_id}")

    try:
        features = get_online_features(FEATURE_STORE, entity_id)
        logger.debug(f"Feature values fetched from Feast: {features}")

    except Exception as e:
        logger.error(f"Feast lookup failed for entity_id={entity_id}: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    # Check for null features
    null_fields = [col for col in MODEL_FEATURE_NAMES if features.get(col) is None]
    if null_fields:
        logger.warning(f"Null features detected for entity_id={entity_id}, fields: {null_fields}")

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
        logger.info(f"Features used for explanation — entity_id={entity_id}")

    logger.info(
        f"Prediction complete — result={prediction}, probability={probability:.4f}, "
        f"version={MODEL_VERSION}"
    )

    return response


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
