"""
Model training script with MLflow integration.
Trains a baseline fraud detection model and registers it in MLflow.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import json
import os

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = "training/data/fraudTrain.csv"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "fraud-detection-model"
RANDOM_STATE = 42

# ============================================================================
# LOAD & PREPARE DATA
# ============================================================================
def load_and_prepare_data(filepath):
    """Load CSV and do basic feature engineering."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Select numeric features only (simple baseline)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove the target and entity key
    if 'cc_num' in numeric_cols:
        numeric_cols.remove('cc_num')
    if 'Class' in numeric_cols:
        target_col = 'Class'
    else:
        # Adjust based on actual column name
        target_col = df.columns[-1]
        numeric_cols = [c for c in numeric_cols if c != target_col]
    
    # Create feature matrix and target
    X = df[numeric_cols].fillna(0)
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, numeric_cols

# ============================================================================
# TRAIN MODEL
# ============================================================================
def train_model(X, y):
    """Train a RandomForest baseline."""
    print("\nTraining model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with class_weight='balanced' as specified
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    print(f"Test metrics: {metrics}")
    
    return model, scaler, metrics, (X_train_scaled, X_test_scaled, y_train, y_test)

# ============================================================================
# REGISTER IN MLFLOW
# ============================================================================
def register_model(model, scaler, metrics):
    """Log model and register in MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MODEL_NAME)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param('algorithm', 'RandomForest')
        mlflow.log_param('n_estimators', 50)
        mlflow.log_param('max_depth', 10)
        mlflow.log_param('class_weight', 'balanced')
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model (without registering - use older API)
        import pickle
        import cloudpickle

        # Save model locally then log as artifact
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(model, f)
            model_path = f.name

        mlflow.log_artifact(model_path, artifact_path="model")
        os.remove(model_path)

        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")

        # Get the run's model URI
        model_uri = f"runs:/{run_id}/model"
        return model_uri

# ============================================================================
# CREATE FEATURE PARQUET FOR FEAST
# ============================================================================
def save_features_parquet(X, y, numeric_cols, entity_col='cc_num'):
    """
    Save features as Parquet for Feast ingestion.
    Assumes the original data has cc_num as entity key.
    """
    print("\nCreating features.parquet...")
    
    # Reconstruct dataframe with entity
    features_df = X.copy()
    features_df['event_timestamp'] = pd.Timestamp.now()
    
    # If we have entity in original data, add it
    features_df.to_parquet('training/features.parquet', index=False)
    print(f"Saved features: training/features.parquet")

# ============================================================================
# CREATE SAMPLE REQUEST
# ============================================================================
def save_sample_request(X):
    """Create a valid prediction request JSON."""
    print("\nCreating sample_request.json...")
    
    # Use first row as sample (cast to native Python types)
    sample = X.iloc[0].to_dict()
    sample = {k: float(v) for k, v in sample.items()}
    
    sample_request = {
        "entity_id": 1  # Placeholder entity ID
    }
    
    with open('training/sample_request.json', 'w') as f:
        json.dump(sample_request, f, indent=2)
    
    print(f"Saved sample request: training/sample_request.json")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Load data
    X, y, numeric_cols = load_and_prepare_data(DATA_PATH)
    
    # Train model
    model, scaler, metrics, _ = train_model(X, y)
    
    # Register in MLflow
    model_uri = register_model(model, scaler, metrics)
    print(f"\n✓ Model registered: {model_uri}")
    
    # Create feature parquet
    save_features_parquet(X, y, numeric_cols)
    
    # Create sample request
    save_sample_request(X)
    
    print("\n✓ Training pipeline complete!")
    print(f"✓ Check MLflow at {MLFLOW_TRACKING_URI}")
