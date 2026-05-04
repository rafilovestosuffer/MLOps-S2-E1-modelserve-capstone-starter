import os
import mlflow
import mlflow.sklearn

from app.logger import get_logger

logger = get_logger(__name__)


def load_production_model():
    """Load the model registered as 'Production' in the MLflow Registry.
    Called once at FastAPI startup — not on every request.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    model_name = os.getenv("MLFLOW_MODEL_NAME", "fraud_detector")

    logger.info(f"Loading model from MLflow Registry: {model_name}@Production")
    logger.info(f"MLflow tracking URI: {mlflow_uri}")

    try:
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)

        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Production"])
        version = versions[0].version if versions else "unknown"
        run_id = versions[0].run_id if versions else "unknown"

        logger.info(f"Model loaded successfully — version: {version}, run_id: {run_id}")
        logger.info(f"Model type: {type(model).__name__}")

        # Check if model version has metrics
        if versions:
            try:
                run = client.get_run(run_id)
                metrics = run.data.metrics
                if not metrics:
                    logger.warning(f"Model version {version} in registry has no metrics logged")
            except Exception as e:
                logger.debug(f"Could not retrieve metrics for model version {version}: {e}")

        return model, version

    except Exception as e:
        logger.exception(f"Failed to load model from MLflow Registry: {e}")
        raise
