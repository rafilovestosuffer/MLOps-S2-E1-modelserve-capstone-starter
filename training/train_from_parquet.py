"""
Train a fraud detection model from features.parquet and register it in MLflow.
Used when fraudTrain.csv is not available locally.
"""
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "fraud_detector"
FEATURE_COLS = ["amt", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"]
RANDOM_STATE = 42

def load_features():
    print("Loading features.parquet...")
    df = pd.read_parquet("training/features.parquet")
    X = df[FEATURE_COLS].fillna(0)

    if "is_fraud" not in df.columns:
        raise ValueError(
            "features.parquet is missing the 'is_fraud' column. "
            "Re-export the parquet from fraudTrain.csv including the label column."
        )
    y = df["is_fraud"].astype(int)
    print(f"Features shape: {X.shape}, fraud rate: {y.mean():.3%}")
    return X, y

def train_and_register(X, y):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=50, max_depth=10,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_proba)),
    }
    print(f"Metrics: {metrics}")

    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run() as run:
        mlflow.log_params({"n_estimators": 50, "max_depth": 10, "class_weight": "balanced"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id

    print(f"Run ID: {run_id}")

    # Set 'champion' alias on the newly registered version
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if versions:
        v = max(versions, key=lambda x: int(x.version)).version
        client.set_registered_model_alias(MODEL_NAME, "champion", v)
        print(f"Model '{MODEL_NAME}' version {v} — alias 'champion' set")
    else:
        print("Warning: could not find version to assign champion alias")

if __name__ == "__main__":
    X, y = load_features()
    train_and_register(X, y)
    print("\nDone! Model registered in MLflow as Production.")
