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
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

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
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, columns: {df.columns.tolist()}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[FEATURE_COLS].fillna(0)
    y = df["is_fraud"]
    print(f"Features: {X.shape}  |  Fraud rate: {y.mean():.3%}")
    return X, y


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(**PARAMS)
    print("Training RandomForest...")
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_proba)),
    }
    print(f"Metrics: {metrics}")
    return model, metrics


def register(model, metrics):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MODEL_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params(PARAMS)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id

    print(f"Run ID: {run_id}")

    # Promote latest version to Production
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    if versions:
        v = versions[0].version
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Model '{MODEL_NAME}' version {v} → Production")
    else:
        print("Warning: could not find version to promote")


if __name__ == "__main__":
    X, y = load_data(DATA_PATH)
    model, metrics = train(X, y)
    register(model, metrics)
    print("\nDone! Model registered in MLflow as Production.")
    print(f"MLflow UI: {MLFLOW_TRACKING_URI}")
