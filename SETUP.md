# ModelServe — Complete Setup & Recovery Guide

## What This Project Does

A fraud detection inference service:
- **FastAPI** at port 8000 — serves predictions
- **MLflow** at port 5000 — model registry (fraud_detector v2, Production)
- **Feast** — feature store using Redis as online store
- **Redis** at port 6379 — stores 1,296,675 feature rows
- **Prometheus metrics** at `/metrics`

---

## File Map

```
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI: /health /predict /metrics /predict/{id}
│   ├── metrics.py           # Prometheus: counters, histogram, gauge
│   ├── model_loader.py      # Loads fraud_detector Production from MLflow
│   ├── feature_client.py    # Feast SDK online feature lookup
│   └── tests/
│       └── test_predict.py  # 5 pytest tests (all deps mocked)
├── feast_repo/
│   ├── feature_store.yaml   # project=modelserve, redis:6379, local offline
│   ├── feature_definitions.py  # cc_num entity, fraud_features view
│   └── data/registry.db    # Feast registry (committed)
├── training/
│   ├── features.parquet     # 1,296,675 rows; cols: cc_num, amt, lat, long,
│   │                        # city_pop, unix_time, merch_lat, merch_long,
│   │                        # event_timestamp (UTC), zip
│   ├── train_from_parquet.py  # Train RF from parquet, register in MLflow
│   ├── train.py             # Original train script (needs fraudTrain.csv)
│   └── sample_request.json  # {"entity_id": 0}
├── Dockerfile               # python:3.10-slim, HEALTHCHECK, uvicorn
├── docker-compose.yml       # postgres, redis, mlflow, api
├── requirements.txt         # Pinned: fastapi==0.111.0, mlflow==2.13.0,
│                            # feast[redis]==0.38.0, scikit-learn==1.4.2, etc.
└── .dockerignore
```

---

## Full Bootstrap (Start From Zero)

### Prerequisites
- Docker Desktop running

### Step 1 — Start Postgres + Redis
```bash
cd MLOps-S2-E1-modelserve-capstone-starter
docker compose up postgres redis -d
```
Wait ~10s for healthchecks.

### Step 2 — Start MLflow
```bash
docker compose up mlflow -d
# Verify it's healthy:
docker compose ps mlflow
# Should show: (healthy)
# Also check: curl http://localhost:5000/health  → OK
```

### Step 3 — Train model and register in MLflow (Production)
```bash
docker run --rm \
  -v "${PWD}:/app" -w /app \
  -v mlops-s2-e1-modelserve-capstone-starter_mlflow_data:/mlflow \
  --network mlops-s2-e1-modelserve-capstone-starter_default \
  -e MLFLOW_TRACKING_URI=http://modelserve-mlflow:5000 \
  python:3.10-slim \
  bash -c "pip install mlflow==2.13.0 scikit-learn==1.4.2 pyarrow pandas -q && GIT_PYTHON_REFRESH=quiet python training/train_from_parquet.py"
```
Expected output: `Model 'fraud_detector' version N promoted to Production`

**IMPORTANT:** The `-v mlflow_data:/mlflow` mount is required so artifacts persist in the Docker volume. Without it, the model registry entry is created but artifacts are missing and the API will fail to load.

### Step 4 — Apply Feast schema and materialize features into Redis
```bash
docker run --rm \
  -v "${PWD}:/app" -w /app \
  --network mlops-s2-e1-modelserve-capstone-starter_default \
  -e FEAST_REDIS_HOST=modelserve-redis \
  python:3.10-slim \
  bash -c "pip install 'feast[redis]==0.38.0' pyarrow pandas -q && cd feast_repo && feast apply && feast materialize-incremental 2026-12-31T23:59:59"
```
Expected: 1,296,675 rows pushed to Redis (~6 minutes).

### Step 5 — Build and start the API
```bash
docker compose up api --build -d
docker compose logs api -f
# Wait for: "Application startup complete."
# Wait for: (healthy) in docker compose ps api
```

### Step 6 — Verify everything
```bash
# Health
curl http://localhost:8000/health
# → {"status":"healthy","model_version":"2"}

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"entity_id": 0}'
# → {"prediction":0,"probability":...,"model_version":"2","timestamp":"..."}

# Explain (shows feature values used)
curl "http://localhost:8000/predict/0?explain=true"
# → includes "features_used":{"amt":4.97,"lat":36.07,...}

# Prometheus metrics
curl http://localhost:8000/metrics | grep prediction
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{"status":"healthy","model_version":"2"}` |
| POST | `/predict` | Body: `{"entity_id": <int>}` → prediction |
| GET | `/predict/{id}` | Same as POST, optionally `?explain=true` |
| GET | `/metrics` | Prometheus metrics |

---

## Key Environment Variables (set in docker-compose.yml)

| Variable | Value | Used by |
|----------|-------|---------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | api |
| `MLFLOW_MODEL_NAME` | `fraud_detector` | api |
| `FEAST_REPO_PATH` | `/app/feast_repo` | api |
| `REDIS_URL` | `redis://redis:6379` | api |

---

## Troubleshooting

### API fails with "No such file or directory: /mlflow/artifacts/..."
The training was run without the `mlflow_data` volume mounted. Re-run Step 3 with the `-v mlflow_data:/mlflow` flag.

### Feast returns null features
1. Check Redis has data: `docker exec modelserve-redis redis-cli DBSIZE` → should be ~1296675
2. If 0, re-run Step 4 (materialize)
3. If non-zero, check Feast key format matches (Feast 0.38 uses plain feature names, no view prefix)

### MLflow container unhealthy
The mlflow image doesn't have `curl`. The healthcheck uses Python:
```yaml
test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
```

### Port conflicts
```bash
# Stop everything
docker compose down
# Start fresh
docker compose up -d
```

---

## Model Details

- **Algorithm:** RandomForestClassifier (n_estimators=50, max_depth=10, class_weight=balanced)
- **Features:** amt, lat, long, city_pop, unix_time, merch_lat, merch_long
- **Entity key:** cc_num (integer row index from features.parquet)
- **Training data:** features.parquet (1,296,675 rows, synthetic ~0.14% fraud labels)
- **Metrics:** precision=0.14, recall=0.99, f1=0.25, ROC-AUC=0.996
- **MLflow experiment:** fraud_detector
- **Registered as:** fraud_detector, stage=Production, version=2

---

## Prometheus Metrics Exposed

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_requests_total` | Counter (labels: status) | Total requests (success/error) |
| `prediction_duration_seconds` | Histogram | Request latency |
| `prediction_errors_total` | Counter | Failed predictions |
| `model_version_info` | Gauge (labels: version) | Current model version |
| `feast_lookup_total` | Counter (labels: result) | Feature lookups (hit/miss) |
