import os
import mlflow
import mlflow.sklearn
import logging

logger = logging.getLogger(__name__)


def load_production_model():
    """Load the model registered as 'Production' in the MLflow Registry.
    Called once at FastAPI startup — not on every request.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    model_name = os.getenv("MLFLOW_MODEL_NAME", "fraud_detector")

    logger.info(f"Loading model '{model_name}' (Production) from {mlflow_uri}")

    try:
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)

        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Production"])
        version = versions[0].version if versions else "unknown"

        logger.info(f"Model loaded successfully — version {version}")
        return model, version

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
