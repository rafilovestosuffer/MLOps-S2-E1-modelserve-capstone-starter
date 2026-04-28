"""
Materialize features from offline parquet store into Redis online store.
Run this after `feast apply` whenever features.parquet is updated.
"""
import os
import sys
import logging
from datetime import datetime, timezone
from feast import FeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_PATH = os.getenv("FEAST_REPO_PATH", os.path.join(os.path.dirname(__file__), "..", "feast_repo"))


def materialize():
    logger.info(f"Connecting to feature store at {REPO_PATH}")
    store = FeatureStore(repo_path=REPO_PATH)

    end_date = datetime.now(tz=timezone.utc)
    logger.info(f"Materializing features up to {end_date.isoformat()}")

    try:
        store.materialize_incremental(end_date=end_date)
        logger.info("Materialization complete.")
    except Exception as e:
        logger.error(f"Materialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    materialize()
