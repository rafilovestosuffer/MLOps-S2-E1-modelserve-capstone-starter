import os
import logging
from feast import FeatureStore

logger = logging.getLogger(__name__)

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
    return FeatureStore(repo_path=repo_path)


def get_online_features(store: FeatureStore, entity_id: int) -> dict:
    """Fetch features for a single cc_num from the Redis online store."""
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

        return features

    except Exception as e:
        logger.error(f"Feast lookup failed for cc_num={entity_id}: {e}")
        raise
