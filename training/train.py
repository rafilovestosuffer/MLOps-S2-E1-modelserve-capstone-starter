"""
ModelServe — Model Training Script
Trains a fraud detection RandomForest model on fraudTrain.csv and registers
it in the MLflow Model Registry with stage 'Production'.

Usage:
    python training/train.py

Requirements:
    - fraudTrain.csv in training/data/fraudTrain.csv
    - MLflow tracking server running (MLFLOW_TRACKING_URI env var)

Reproducibility: running this script again with the same data produces
a functionally equivalent model (same hyperparameters, same random seed).
"""

import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH           = "training/data/fraudTrain.csv"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = "fraud_detector"   # MUST match model_loader.py
RANDOM_STATE        = 42

# These 7 features MUST match feature_client.py and feature_definitions.py
FEATURE_COLS = ["amt", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"]

PARAMS = {
    "n_estimators":  50,
    "max_depth":     10,
    "class_weight":  "balanced",
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}


def load_data(path: str):
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded — shape: {df.shape}, columns: {len(df.columns)}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.error(f"Missing columns in dataset: {missing}")
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[FEATURE_COLS].fillna(0)
    y = df["is_fraud"]

    fraud_count = int(y.sum())
    normal_count = int(len(y) - fraud_count)
    logger.info(f"Class distribution — fraud: {fraud_count}, normal: {normal_count}")
    logger.info(f"Features shape: {X.shape}")

    return X, y


def train(X, y):
    logger.info("Starting fraud detection model training")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    logger.info(f"Training {RandomForestClassifier.__name__} with class_weight='balanced'")
    model = RandomForestClassifier(**PARAMS)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_proba)),
    }

    logger.info(f"Training complete — ROC AUC: {metrics['roc_auc']:.4f}")
    logger.debug(f"Full metrics: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")

    return model, metrics


def register(model, metrics):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MODEL_NAME)

    try:
        with mlflow.start_run() as run:
            mlflow.log_params(PARAMS)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )
            run_id = run.info.run_id

        logger.info(f"Logging metrics and params to MLflow run: {run_id}")

        # Promote latest version to Production
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
        if versions:
            v = versions[0].version
            logger.info(f"Registering model as '{MODEL_NAME}' in MLflow Registry")
            logger.info(f"Model registered — version: {v}")

            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=v,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info(f"Alias 'Production' assigned to version {v}")
        else:
            logger.warning("Could not find model version to promote to Production")

    except Exception as e:
        logger.exception(f"Error during model registration: {e}")
        raise


if __name__ == "__main__":
    try:
        X, y = load_data(DATA_PATH)
        model, metrics = train(X, y)
        register(model, metrics)
        logger.info("Training pipeline completed successfully — model registered in MLflow as Production")
        logger.info(f"MLflow UI: {MLFLOW_TRACKING_URI}")
    except Exception as e:
        logger.exception("Training pipeline failed")
        sys.exit(1)
