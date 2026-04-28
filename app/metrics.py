from prometheus_client import Counter, Histogram, Gauge

prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["status"],
)

prediction_duration_seconds = Histogram(
    "prediction_duration_seconds",
    "Time spent processing a prediction request",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)

model_version_info = Gauge(
    "model_version_info",
    "Currently loaded model version",
    ["version"],
)

feast_lookup_total = Counter(
    "feast_lookup_total",
    "Total Feast feature lookups",
    ["result"],
)
