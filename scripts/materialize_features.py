"""
Materialize features from offline parquet store into Redis online store.
Run this after `feast apply` whenever features.parquet is updated or Redis is cleared.

Pass --full to force a full rematerialization from epoch (e.g. after Redis restart).
Without the flag, uses materialize_incremental (only new rows since last run).
"""
import os
import sys
import argparse
import logging
from datetime import datetime, timezone, timedelta
from feast import FeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_PATH = os.getenv("FEAST_REPO_PATH", os.path.join(os.path.dirname(__file__), "..", "feast_repo"))

# Epoch start that predates all rows in features.parquet (Kaggle fraud dataset ~2019-2020)
FULL_START_DATE = datetime(2019, 1, 1, tzinfo=timezone.utc)


def materialize(full: bool = False):
    logger.info(f"Connecting to feature store at {REPO_PATH}")
    store = FeatureStore(repo_path=REPO_PATH)

    end_date = datetime.now(tz=timezone.utc)

    try:
        if full:
            logger.info(f"Full materialization from {FULL_START_DATE.isoformat()} to {end_date.isoformat()}")
            store.materialize(start_date=FULL_START_DATE, end_date=end_date)
        else:
            logger.info(f"Incremental materialization up to {end_date.isoformat()}")
            store.materialize_incremental(end_date=end_date)
        logger.info("Materialization complete.")
    except Exception as e:
        logger.error(f"Materialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true",
        help="Force full rematerialization from epoch (use after Redis restart or first run)",
    )
    args = parser.parse_args()
    materialize(full=args.full)
