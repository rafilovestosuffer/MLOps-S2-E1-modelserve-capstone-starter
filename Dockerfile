# ============================================================================
# ModelServe — Multi-Stage Dockerfile
# Stage 1 (builder): install Python dependencies
# Stage 2 (runtime): minimal image, non-root user, production ASGI server
# Target image size: < 800 MB
# ============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.10-slim@sha256:db7a1753878f2e564b6a0257052d3fa9aeb28ceb35ea82ff6303939b5c1e6de2 AS builder

WORKDIR /build

# Install deps into an isolated prefix so we can copy them cleanly
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim@sha256:db7a1753878f2e564b6a0257052d3fa9aeb28ceb35ea82ff6303939b5c1e6de2 AS runtime

WORKDIR /app

# curl is needed for the HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY feast_repo/ ./feast_repo/
COPY training/ ./training/
COPY scripts/ ./scripts/

# Create non-root user and fix ownership
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
