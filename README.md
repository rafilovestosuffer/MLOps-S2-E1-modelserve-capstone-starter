# ModelServe

A production-grade fraud detection serving platform built for the MLOps with Cloud
Season 2 capstone. Wraps a RandomForest classifier in MLflow experiment tracking,
Feast feature store, FastAPI inference API, Prometheus + Grafana observability,
Pulumi-provisioned AWS infrastructure, and a GitHub Actions CI/CD pipeline.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Docker + Docker Compose plugin | 24+ | Run the full stack locally |
| Python | 3.10 | Model training, Pulumi |
| AWS CLI | 2.x | Interact with AWS resources |
| Pulumi CLI | 3.x | Provision infrastructure |
| Git | any | Version control |

---

## Quick Start — Local Development

```bash
# 1. Clone
git clone https://github.com/rafilovestosuffer/MLOps-S2-E1-modelserve-capstone-starter.git
cd MLOps-S2-E1-modelserve-capstone-starter

# 2. Copy environment file (edit values if needed)
cp .env.example .env

# 3. Start the full stack (postgres, redis, mlflow, api, prometheus, grafana)
docker compose up --build -d

# 4. Wait ~30 seconds for all services to become healthy, then train the model
docker compose run --rm api python training/train_from_parquet.py

# 5. Verify
curl http://localhost:8000/health
# → {"status": "healthy", "model_version": "1"}

curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @training/sample_request.json
# → {"prediction": 0, "probability": 0.12, "model_version": "1", "timestamp": "..."}
```

**Service URLs (local):**

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

---

## REST Endpoints

| Method | Path | Request Body | Response |
|--------|------|-------------|---------|
| GET | `/health` | — | `{"status": "healthy", "model_version": "<v>"}` |
| POST | `/predict` | `{"entity_id": <int>}` | `{"prediction": 0\|1, "probability": <float>, "model_version": "<v>", "timestamp": "<iso>"}` |
| GET | `/predict/<id>?explain=true` | — | prediction + `"features_used": {...}` |
| GET | `/metrics` | — | Prometheus text format |

---

## Environment Variables

All variables are documented in [`.env.example`](.env.example).

| Variable | Default | Service | Description |
|----------|---------|---------|-------------|
| `POSTGRES_USER` | `mlflow` | postgres, mlflow | PostgreSQL username |
| `POSTGRES_PASSWORD` | `mlflow` | postgres, mlflow | PostgreSQL password |
| `POSTGRES_DB` | `mlflow` | postgres, mlflow | Database name |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | api | MLflow server URL |
| `MLFLOW_MODEL_NAME` | `fraud_detector` | api | Registered model name |
| `MLFLOW_ARTIFACT_ROOT` | `/mlflow/artifacts` | mlflow | Artifact storage path (use `s3://...` in production) |
| `FEAST_REPO_PATH` | `/app/feast_repo` | api | Feast feature store repo path |
| `REDIS_URL` | `redis://redis:6379` | api | Redis connection URL (Feast online store) |
| `GF_SECURITY_ADMIN_USER` | `admin` | grafana | Grafana admin username |
| `GF_SECURITY_ADMIN_PASSWORD` | `admin` | grafana | Grafana admin password |
| `AWS_ACCESS_KEY_ID` | — | Pulumi, MLflow (S3) | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | — | Pulumi, MLflow (S3) | AWS secret key |
| `AWS_DEFAULT_REGION` | `ap-southeast-1` | Pulumi, AWS CLI | AWS region |

---

## GitHub Secrets (required for CI/CD)

Set these in your repository Settings → Secrets and variables → Actions:

| Secret | Purpose |
|--------|---------|
| `AWS_ACCESS_KEY_ID` | AWS credential for Pulumi, ECR push, and EC2 deploy |
| `AWS_SECRET_ACCESS_KEY` | AWS credential (secret key) |
| `PULUMI_ACCESS_TOKEN` | Pulumi Cloud state backend token |
| `SSH_PUBLIC_KEY` | EC2 key pair public key (Pulumi creates the key pair from this) |
| `SSH_PRIVATE_KEY` | EC2 SSH private key (deploy job uses this to SSH in) |
| `DEPLOY_HOST` | EC2 Elastic IP address — set this after the first `pulumi up` |

---

## Deploying to AWS

```bash
# From the Poridhi VM (or any machine with AWS credentials):

export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export SSH_PUBLIC_KEY=$(cat ~/.ssh/modelserve.pub)

cd infrastructure
pip install -r requirements.txt
pulumi login --local          # or: pulumi login (cloud backend)
pulumi stack init prod
pulumi config set aws:region ap-southeast-1
pulumi up --yes

# Get the Elastic IP for the DEPLOY_HOST secret:
pulumi stack output instance_ip
```

**After `pulumi up` completes, wait ~5–8 minutes** for the EC2 user-data bootstrap
to finish (Docker install + repo clone + model training). Then:

```bash
EC2_IP=$(pulumi stack output instance_ip)
curl http://$EC2_IP:8000/health   # should return 200
```

---

## Teardown

**Run this at the end of every Poridhi VM session:**

```bash
git push origin main              # save work first!
cd infrastructure
pulumi destroy --yes
```

---

## Running Tests

```bash
pip install pytest httpx fastapi "feast[redis]==0.38.0" mlflow scikit-learn prometheus-client numpy pandas pydantic
pytest app/tests/ -v
```

---

## Engineering Documentation

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full architecture
documentation, including:
- System overview and design philosophy
- Architecture diagrams (local and production topologies)
- Five Architecture Decision Records (ADRs)
- CI/CD pipeline documentation
- Operations runbook
- Known limitations

---

## Dataset

[Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
— Simulated credit card transactions (Sparkov). Training uses `fraudTrain.csv`
(~1.3 M rows). Entity key: `cc_num`. A pre-processed `features.parquet` is committed
to the repository so EC2 bootstrap does not require the full CSV.

---

*MLOps with Cloud Season 2 — Capstone: ModelServe | Poridhi.io*
