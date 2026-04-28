# ModelServe — Engineering Documentation

> **Graded deliverable (19 marks).** This document describes the architecture,
> design decisions, CI/CD pipeline, and operational procedures for ModelServe —
> a production-grade ML serving platform built for the MLOps with Cloud Season 2 capstone.

---

## 1. System Overview

ModelServe is a production-grade fraud detection serving platform that wraps a trained
RandomForest classifier in the full MLOps infrastructure required to serve it reliably.
The system accepts a credit card number (`entity_id`) via a REST API, fetches the
card's transaction features from a Redis-backed feature store (Feast), runs inference
against the latest registered model from an MLflow Model Registry, and returns a
structured prediction with probability, model version, and timestamp. Every request
is instrumented with Prometheus metrics, and a Grafana dashboard provides real-time
visibility into latency, error rate, and feature store health.

The design philosophy is **simplicity over complexity**. All components run on a single
AWS EC2 instance (`t3.small`) in the `ap-southeast-1` region, orchestrated by Docker
Compose. This topology eliminates cross-host networking, simplifies debugging, and
makes the end-to-end system reproducible from a single `docker compose up` command.
It is a deliberate trade-off: a single point of failure is acceptable for a capstone
project where demo reliability matters more than high availability.

The infrastructure is provisioned as code with Pulumi (Python), deployed automatically
by a four-job GitHub Actions pipeline, and can be torn down completely with
`pulumi destroy --yes`. The engineering documentation, ADRs, and runbook are treated
as first-class deliverables alongside the code — the system is only as good as the
documentation that explains why it was built the way it is.

---

## 2. Architecture Diagram

### 2.1 Production Topology (EC2)

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  GitHub Actions (CI/CD)                                                 │
  │  ┌──────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐    │
  │  │  test    │→ │infrastructure│→ │build-and-push│→ │   deploy     │    │
  │  │  (pytest)│  │(pulumi up)   │  │(Docker→ECR) │  │(SSH→EC2)    │    │
  │  └──────────┘  └──────────────┘  └─────────────┘  └──────────────┘    │
  └─────────────────────────┬───────────────────────────────────────────────┘
                            │ git push / pulumi up
                            ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  AWS ap-southeast-1                                                     │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │  ECR: modelserve-api  (Docker image registry)                   │    │
  │  └─────────────────────────────────────────────────────────────────┘    │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │  S3: modelserve-artifacts-*  (MLflow artifact store)            │    │
  │  └─────────────────────────────────────────────────────────────────┘    │
  │                                                                         │
  │  VPC 10.0.0.0/16  │  Public Subnet 10.0.1.0/24                         │
  │  ┌──────────────────────────────────────────────────────────────────┐   │
  │  │  EC2: t3.small  (Elastic IP — stable public address)            │   │
  │  │                                                                  │   │
  │  │  ┌─────────────────┐   :8000   TA / curl / browser              │   │
  │  │  │  FastAPI (api)  │◄──────────────────────────────────────     │   │
  │  │  │  uvicorn        │                                             │   │
  │  │  └────┬──────┬─────┘                                            │   │
  │  │       │      │                                                   │   │
  │  │  Feast SDK   MLflow SDK                                          │   │
  │  │       │      │                                                   │   │
  │  │  ┌────▼──┐  ┌▼──────────────────────────────────────────────┐  │   │
  │  │  │ Redis │  │  MLflow Server  :5000                          │  │   │
  │  │  │ :6379 │  │  backend: PostgreSQL :5432                    │  │   │
  │  │  │(Feast │  │  artifacts:  S3 (production)                  │  │   │
  │  │  │ online│  │              /mlflow/artifacts (local dev)    │  │   │
  │  │  │ store)│  └───────────────────────────────────────────────┘  │   │
  │  │  └───────┘                                                       │   │
  │  │                                                                  │   │
  │  │  ┌────────────────────┐  ┌────────────────────────────────────┐ │   │
  │  │  │ Prometheus  :9090  │  │  Grafana  :3000                    │ │   │
  │  │  │ scrapes /metrics   │  │  provisioned dashboard + alerts    │ │   │
  │  │  └────────────────────┘  └────────────────────────────────────┘ │   │
  │  │                                                                  │   │
  │  │  ┌─────────────────────────────────────────────┐                │   │
  │  │  │  PostgreSQL  :5432  (MLflow backend store)  │                │   │
  │  │  └─────────────────────────────────────────────┘                │   │
  │  └──────────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Local Development Topology

```
  Poridhi VM / Developer Laptop
  ┌──────────────────────────────────────────────────────┐
  │  docker compose up                                   │
  │                                                      │
  │  FastAPI :8000  ←→  Redis :6379  (Feast online)     │
  │  FastAPI :8000  ←→  MLflow :5000 (model loading)    │
  │  MLflow  :5000  ←→  PostgreSQL :5432 (backend store)│
  │  MLflow  :5000  ←→  /mlflow/artifacts (local volume)│
  │  Prometheus :9090  scrapes  FastAPI :8000/metrics    │
  │  Grafana :3000  reads  Prometheus :9090              │
  └──────────────────────────────────────────────────────┘
```

**Port summary:**

| Port | Service | Accessible from |
|------|---------|----------------|
| 8000 | FastAPI inference API | Public (TA demo) |
| 5000 | MLflow tracking server | Public (TA demo) |
| 9090 | Prometheus | Public (TA demo) |
| 3000 | Grafana | Public (TA demo) |
| 5432 | PostgreSQL | Internal only |
| 6379 | Redis | Internal only |

---

## 3. Architecture Decision Records (ADRs)

### ADR-1: Deployment Topology

**Context:** The exam allows any deployment topology: everything on EC2, everything
on the Poridhi VM, or a hybrid split. The Poridhi VM state is reset at the end of
each session, making it unreliable as a long-term deployment target.

**Decision:** Option A — all services run on a single AWS EC2 `t3.small` instance
in `ap-southeast-1`, provisioned by Pulumi, with an Elastic IP for a stable address.

**Rationale:** The Poridhi VM does not persist between sessions, so any deployment
there requires a full re-bootstrap at the start of each session. EC2 persists for the
lifetime of the Pulumi stack, giving a stable IP for TA demo purposes. A single-node
topology also eliminates cross-host networking complexity, making the system easier
to debug under demo pressure.

**Trade-offs:** Single point of failure — if the EC2 instance crashes, everything goes
down. No horizontal scaling. Resource contention between services on a `t3.small`
(2 vCPU, 2 GB RAM) could cause memory pressure under heavy load. In a real production
deployment, stateful services (PostgreSQL, Redis, MLflow) would be separated onto
managed services (RDS, ElastiCache, a dedicated MLflow server).

---

### ADR-2: CI/CD Strategy — Incremental Update

**Context:** The GitHub Actions pipeline must deploy on every push to `main`. The
two main strategies are: (a) destroy-and-recreate infrastructure on every push, or
(b) keep the EC2 instance running and only update the API container.

**Decision:** Incremental update. The EC2 instance is persistent. On each push, the
GitHub Actions pipeline runs `pulumi up` (which is idempotent and only changes what
changed), builds and pushes a new Docker image to ECR, then SSHes into EC2 and does
`docker compose pull api && docker compose up api -d --force-recreate`.

**Rationale:** Destroy-and-recreate would take 5–10 minutes per push (EC2 boot +
Docker install + `git clone` + model training). It also destroys the MLflow model
registry on every deploy, requiring re-training. The incremental approach keeps the
MLflow state intact and reduces deploy time to under 2 minutes.

**Trade-offs:** If infrastructure drift occurs (someone manually changes an EC2 setting
outside Pulumi), `pulumi up` may not fully reconcile it. In practice, for this capstone,
all infrastructure changes go through Pulumi, so drift is unlikely. The model registry
is not re-trained on every deploy, which means a stale model could serve traffic after
code changes — acceptable for a demo system, not for production.

---

### ADR-3: Data Architecture

**Context:** The system needs three data stores: an experiment tracking backend (for
MLflow metadata), an artifact store (for model files), and an online feature store
(for low-latency inference).

**Decision:**
- **MLflow backend store**: PostgreSQL (running in Docker on EC2)
- **MLflow artifact store**: S3 bucket (`s3://modelserve-artifacts-*/mlflow-artifacts`) in production; local `/mlflow/artifacts` Docker volume in local development
- **Feast online store**: Redis (running in Docker on EC2)
- **Feast offline store**: local `features.parquet` file (committed to the repository)

**Rationale:** PostgreSQL is the most reliable SQL backend for MLflow — it handles
concurrent writes from multiple training runs correctly, which SQLite does not. S3
for artifacts means the trained model survives a container restart or EC2 stop/start;
the model is not lost when the container is replaced. Redis provides sub-millisecond
feature lookups, which is essential for low-latency inference — the alternative
(querying PostgreSQL or a file directly on each request) would add 10–100ms per call.
The offline store uses parquet files because the dataset is small (~150 KB) and
already committed; migrating to S3-backed offline storage would not improve
inference performance.

**Trade-offs:** Running PostgreSQL as a Docker container means its data lives on a
Docker volume — if the EC2 instance is terminated (not just stopped), MLflow
experiment history is lost. In production, this would be Amazon RDS with automated
backups. Redis also has no persistence enabled, so a Redis container restart loses
materialized features and requires re-running `feast materialize-incremental`. S3
artifact storage requires AWS credentials at training time and adds ~50ms latency
to model loading at API startup.

---

### ADR-4: Containerization

**Context:** The FastAPI service needs a Docker image that is small enough to push/pull
quickly, runs as a non-root user for security, and uses a production-grade ASGI server.

**Decision:** Multi-stage Dockerfile. Stage 1 (`builder`): `python:3.10-slim` installs
all Python dependencies into `/install`. Stage 2 (`runtime`): fresh `python:3.10-slim`
copies only the installed packages and application source, creates a non-root user
(`appuser`, UID 1000), and runs `uvicorn` as the ASGI server.

**Rationale:** Multi-stage builds prevent build tools and pip cache from bloating the
final image. `python:3.10-slim` (vs `python:3.10`) reduces the base image size by
~200 MB by omitting dev headers and test tools. The non-root user prevents container
escape vulnerabilities from escalating to host root privileges. `uvicorn` is appropriate
for a single-worker API at this traffic level; for higher concurrency, `gunicorn` with
`uvicorn` workers would be used.

**Trade-offs:** Multi-stage builds are slower to build locally because both stages
must run. The `python:3.10-slim` base does not include system packages needed by
some ML libraries (e.g., `libgomp` for LightGBM, `libpq-dev` for psycopg2-binary) —
these must be explicitly installed in the Dockerfile. The final image is approximately
600–700 MB due to scikit-learn, MLflow, and Feast dependencies, which is under the
800 MB exam requirement but still large by application standards.

---

### ADR-5: Monitoring Design

**Context:** The system needs Prometheus metrics, Grafana dashboards, and alert rules
that satisfy the exam requirements: latency p50/p95/p99, request rate, error rate,
model version, Feast hit/miss ratio, and at least three alert rules.

**Decision:**
- **Metrics**: `prediction_requests_total` (counter, labeled by status), `prediction_duration_seconds` (histogram with 9 buckets from 5ms to 2.5s), `prediction_errors_total` (counter), `model_version_info` (gauge), `feast_lookup_total` (counter, labeled by result hit/miss)
- **Alert thresholds**: ServiceDown (API down for 30s), HighLatency (p95 > 500ms for 2 min), HighErrorRate (error rate > 5% for 2 min), LowFeastHitRate (hit rate < 80% for 5 min)
- **Grafana**: Fully provisioned from `monitoring/grafana/provisioning/` — no manual UI setup required after `docker compose up`

**Rationale:** The p95 latency threshold of 500ms is appropriate for a fraud detection
API — below that feels snappy; above that users and upstream systems notice. The 5%
error rate threshold gives a signal before errors become pervasive. The Feast hit rate
threshold of 80% catches Redis cache misses early, which usually indicate a failed
feature materialization rather than a traffic spike.

**Trade-offs:** Alert thresholds are not calibrated against real traffic — they were
chosen to be demonstrable during the TA demo with synthetic load. The 500ms p95
threshold will fire easily under a sustained load test, which is intentional for demo
purposes. The `model_version_info` gauge does not track model performance over time
(accuracy drift, data drift) — this would require a separate offline evaluation
pipeline beyond the scope of this capstone. Grafana alerting is not configured
(Prometheus AlertManager handles the alerts), so alert notifications do not reach
any external channel.

---

## 4. CI/CD Pipeline Documentation

### Overview

The GitHub Actions workflow in `.github/workflows/deploy.yml` has four jobs:

```
push to main
    │
    ▼
[1] test ─────────────────────────────── runs on every push and PR
    │ (passes)
    ▼
[2] infrastructure ───────────────────── runs on push to main only
    │ (pulumi up — idempotent)
    ▼
[3] build-and-push ───────────────────── runs on push to main only
    │ (Docker build → ECR push)
    ▼
[4] deploy ───────────────────────────── runs on push to main only
       (SSH to EC2 → pull image → restart api → verify /health)
```

### Job Details

**Job 1: test**
- Trigger: every push, every pull request to `main`
- Runtime: `ubuntu-latest`, Python 3.10
- Actions: `pip install` all API dependencies, `pytest app/tests/ -v`
- Failure behaviour: blocks all downstream jobs; no deploy happens
- Duration: ~90 seconds

**Job 2: infrastructure**
- Trigger: push to `main` only, after `test` passes
- Runtime: `ubuntu-latest`, Python 3.10
- Actions: `pulumi up --yes` with `--stack prod` using the `pulumi/actions@v5` action
- Strategy: incremental update (see ADR-2)
- Secrets required: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `PULUMI_ACCESS_TOKEN`, `SSH_PUBLIC_KEY`
- Failure behaviour: blocks `build-and-push` and `deploy`
- Duration: ~60 seconds (no-op if infrastructure unchanged)

**Job 3: build-and-push**
- Trigger: push to `main` only, after `infrastructure` passes
- Runtime: `ubuntu-latest`
- Actions: ECR login → `docker build` → `docker push` (both SHA tag and `latest`)
- Secrets required: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- Failure behaviour: blocks `deploy`
- Duration: ~3–5 minutes (Docker build)

**Job 4: deploy**
- Trigger: push to `main` only, after `build-and-push` passes
- Runtime: `ubuntu-latest`
- Actions: SSH to EC2 via `appleboy/ssh-action@v1` → `git pull` → ECR login → `docker compose pull api` → `docker compose up api -d --force-recreate`
- Post-deploy verification: polls `http://<DEPLOY_HOST>:8000/health` up to 12 times with 10s sleep between attempts
- Secrets required: `SSH_PRIVATE_KEY`, `DEPLOY_HOST`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- Failure behaviour: marks the workflow run as failed; previous version continues serving
- Duration: ~60–90 seconds

### Required GitHub Secrets

| Secret | Purpose |
|--------|---------|
| `AWS_ACCESS_KEY_ID` | AWS credential for Pulumi, ECR, and EC2 deploy |
| `AWS_SECRET_ACCESS_KEY` | AWS credential (secret key) |
| `PULUMI_ACCESS_TOKEN` | Authenticates to Pulumi Cloud state backend |
| `SSH_PUBLIC_KEY` | EC2 key pair public key (Pulumi uses this to create the key pair) |
| `SSH_PRIVATE_KEY` | EC2 SSH private key (deploy job uses this to SSH in) |
| `DEPLOY_HOST` | EC2 Elastic IP address (set after first `pulumi up`) |

### Expected End-to-End Deploy Time

| Phase | Duration |
|-------|---------|
| test | ~90s |
| infrastructure (no changes) | ~60s |
| build-and-push | ~3–5 min |
| deploy + health check | ~90s |
| **Total** | **~6–8 min** |

---

## 5. Runbook

### 5.1 Bootstrapping from a Fresh Clone (Poridhi VM)

Follow these steps at the start of a new Poridhi VM session:

```bash
# Step 1 — Set AWS credentials (from Poridhi sandbox dashboard)
export AWS_ACCESS_KEY_ID=<paste from dashboard>
export AWS_SECRET_ACCESS_KEY=<paste from dashboard>
export AWS_DEFAULT_REGION=ap-southeast-1

# Step 2 — Clone the repository
git clone https://github.com/rafilovestosuffer/MLOps-S2-E1-modelserve-capstone-starter.git
cd MLOps-S2-E1-modelserve-capstone-starter

# Step 3 — Generate SSH key for EC2 (skip if you have the key from a previous session)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/modelserve -N ""
export SSH_PUBLIC_KEY=$(cat ~/.ssh/modelserve.pub)

# Step 4 — Install Pulumi Python dependencies
cd infrastructure
pip install -r requirements.txt

# Step 5 — Login to Pulumi (use --local for offline state, or use token for cloud)
pulumi login --local
# OR: pulumi login   (requires PULUMI_ACCESS_TOKEN env var or browser login)

# Step 6 — Initialize stack and set region
pulumi stack init prod
pulumi config set aws:region ap-southeast-1

# Step 7 — Deploy infrastructure (~3–5 min for EC2 boot)
pulumi up --yes

# Step 8 — Get EC2 IP
export EC2_IP=$(pulumi stack output instance_ip)
echo "EC2 IP: $EC2_IP"

# Step 9 — Set GitHub Secrets (requires gh CLI and GitHub authentication)
gh secret set AWS_ACCESS_KEY_ID --body "$AWS_ACCESS_KEY_ID"
gh secret set AWS_SECRET_ACCESS_KEY --body "$AWS_SECRET_ACCESS_KEY"
gh secret set DEPLOY_HOST --body "$EC2_IP"
gh secret set SSH_PRIVATE_KEY < ~/.ssh/modelserve
# Set PULUMI_ACCESS_TOKEN via: gh secret set PULUMI_ACCESS_TOKEN

# Step 10 — Wait ~5 min for EC2 user-data to complete, then verify
curl http://$EC2_IP:8000/health
# Expected: {"status": "healthy", "model_version": "1"}

# Step 11 — Push code to trigger full CI/CD pipeline
cd ..
git push origin main
```

### 5.2 Deploying a New Model Version

To retrain the model and deploy without restarting the whole stack:

```bash
# Option A — Retrain on EC2 directly (no CSV needed, uses features.parquet)
ssh -i ~/.ssh/modelserve ubuntu@$EC2_IP
cd ~/modelserve
docker compose run --rm api python training/train_from_parquet.py
# The new model version is automatically promoted to Production in MLflow.
# Restart the API container to load the new version:
docker compose restart api

# Option B — Retrain locally (requires fraudTrain.csv) then push
python training/train.py
git add training/   # if you want to commit updated features.parquet
git push origin main  # CI/CD pipeline redeploys the API automatically
```

### 5.3 Common Failure Recovery

**API container keeps restarting (OOMKilled or crash loop):**
```bash
ssh -i ~/.ssh/modelserve ubuntu@$EC2_IP
cd ~/modelserve
docker compose logs api --tail=50   # read the error
docker compose up api -d            # restart
```

**MLflow cannot connect to PostgreSQL (backend store error):**
```bash
docker compose logs mlflow --tail=30
docker compose logs postgres --tail=20
docker compose restart postgres mlflow
# If postgres volume is corrupted:
docker compose down
docker volume rm modelserve_postgres_data
docker compose up -d postgres
# Then wait for postgres health, then start mlflow, then re-train model
```

**S3 permission loss (IAM role issue):**
```bash
# On EC2, test S3 access:
aws s3 ls s3://$(cat .env | grep MLFLOW_ARTIFACT_ROOT | cut -d/ -f3)
# If access denied: check the IAM instance profile in AWS console
# Re-run pulumi up from the Poridhi VM to re-apply IAM policies
cd ~/modelserve/infrastructure && pulumi up --yes
```

**Pulumi state corruption (state lock or conflict):**
```bash
cd infrastructure
pulumi stack export > backup.json   # backup before fixing
# If state is locked:
pulumi cancel
# If serious corruption, restore from backup:
pulumi stack import < backup.json
```

**Redis data loss (Feast cache empty, all feature lookups miss):**
```bash
ssh -i ~/.ssh/modelserve ubuntu@$EC2_IP
cd ~/modelserve
docker compose restart redis
# Re-materialize features into Redis:
docker compose run --rm api feast -c feast_repo materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
```

**GitHub Actions failing at deploy step (SSH connection refused):**
```bash
# Check that DEPLOY_HOST secret has the current EC2 IP
pulumi stack output instance_ip
# Update the GitHub secret if the IP changed after a pulumi up:
gh secret set DEPLOY_HOST --body "$(pulumi stack output instance_ip)"
```

### 5.4 Teardown (End of Session)

**Always run this at the end of every Poridhi VM session:**

```bash
# Step 1 — Push all work to GitHub first
git add -A
git commit -m "End of session: save progress"
git push origin main

# Step 2 — Destroy all AWS resources
cd infrastructure
pulumi destroy --yes

# Step 3 — Verify destruction (should return empty)
aws ec2 describe-instances --filters "Name=tag:Project,Values=modelserve" \
  --query "Reservations[].Instances[].InstanceId" --output text

echo "Done. AWS resources destroyed."
```

---

## 6. Known Limitations

1. **Single point of failure.** All services run on one EC2 instance. A hardware
   failure or OS crash takes everything down. In production, stateful services
   (PostgreSQL, Redis, MLflow) would run on managed AWS services (RDS, ElastiCache,
   a dedicated server) with Multi-AZ replication.

2. **No TLS/HTTPS.** All endpoints are served over plain HTTP. A production deployment
   would put an Application Load Balancer with an ACM certificate in front of the
   FastAPI service.

3. **No authentication on any endpoint.** The `/predict`, `/metrics`, MLflow UI, and
   Grafana endpoints are all publicly accessible. A production system would require
   API keys for `/predict`, restrict `/metrics` to the Prometheus scraper IP, and
   protect the MLflow UI with a VPN or basic auth.

4. **Model state is not durable across EC2 termination.** PostgreSQL and MLflow
   artifacts are stored on Docker volumes. If the EC2 instance is terminated (not just
   stopped), all experiment history is lost. Mitigation: use S3 for MLflow artifacts
   (already done) and RDS for the backend store.

5. **Redis has no persistence.** If the Redis container restarts, all materialized
   features are lost and every subsequent Feast lookup will fail (miss) until
   `feast materialize-incremental` is re-run.

6. **Feature data uses synthetic fraud labels.** The `train_from_parquet.py` script
   generates labels from a simple threshold rule on transaction amount, not real
   fraud labels. The model's ROC AUC reflects this synthetic task, not real-world
   fraud detection quality.

7. **No model performance monitoring.** The Grafana dashboard shows infrastructure
   metrics (latency, error rate) but not model quality metrics (accuracy drift, data
   distribution shift). A production system would run periodic offline evaluation
   and alert on significant performance degradation.

8. **User-data bootstrap takes 5–10 minutes.** After `pulumi up`, the EC2 instance
   takes several minutes to install Docker, clone the repo, and train the model.
   There is no way to know from outside when bootstrap is complete (other than
   polling `/health`). A production setup would use a pre-baked AMI (Packer) to
   reduce cold-start time to under 60 seconds.

9. **`pulumi login --local` state is stored on the Poridhi VM.** If the VM resets
   before `pulumi destroy` is run, the Pulumi state is lost and AWS resources become
   unmanaged. Mitigation: always run `pulumi destroy --yes` before the session ends,
   and use Pulumi Cloud state backend (`pulumi login`) with a token for durability.
