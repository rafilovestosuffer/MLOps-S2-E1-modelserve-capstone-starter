import os
from feast import FeatureStore

from app.logger import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "fraud_features:amt",
    "fraud_features:lat",
    "fraud_features:long",
    "fraud_features:city_pop",
    "fraud_features:unix_time",
    "fraud_features:merch_lat",
    "fraud_features:merch_long",
]

MODEL_FEATURE_NAMES = [f.split(":")[1] for f in FEATURE_COLUMNS]


def get_feature_store():
    repo_path = os.getenv("FEAST_REPO_PATH", "/app/feast_repo")
    store = FeatureStore(repo_path=repo_path)
    # Log initialization on first call (when store.repo_config is available)
    try:
        online_store = getattr(store.repo_config, 'online_store', {})
        online_store_type = type(online_store).__name__
        logger.info(f"Feast feature store initialized — online store: {online_store_type}")
    except Exception as e:
        logger.debug(f"Could not determine online store type: {e}")
    return store


def get_online_features(store: FeatureStore, entity_id: int) -> dict:
    """Fetch features for a single cc_num from the Redis online store."""
    logger.debug(f"Feature lookup for entity_id={entity_id}")

    try:
        entity_rows = [{"cc_num": entity_id}]
        response = store.get_online_features(
            features=FEATURE_COLUMNS,
            entity_rows=entity_rows,
        ).to_dict()

        features = {}
        for col in MODEL_FEATURE_NAMES:
            # Feast 0.38 to_dict() returns plain feature names (no view prefix)
            val = response.get(col, [None])[0]
            features[col] = val

        # Check if features were found (at least one non-null value)
        non_null_features = {k: v for k, v in features.items() if v is not None}
        if non_null_features:
            logger.info(f"Feature cache HIT for entity_id={entity_id}")
            logger.debug(f"Features returned: {features}")
        else:
            logger.warning(f"Feature cache MISS for entity_id={entity_id} — entity not in Redis")

        return features

    except Exception as e:
        logger.error(f"Feast lookup failed for entity_id={entity_id}: {e}")
        raise
