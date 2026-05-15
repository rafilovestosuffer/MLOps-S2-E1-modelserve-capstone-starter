# Logging

All logging uses Python's built-in `logging` module. Configuration is in `app/logger.py`.

## Setup

Import and use:
```python
from app.logger import get_logger
logger = get_logger(__name__)
```

The root logger is configured on startup with `configure_logging()`.

## Log Level

Control verbosity with `LOG_LEVEL` env var (default: INFO):
```bash
export LOG_LEVEL=DEBUG    # Verbose, includes feature values
export LOG_LEVEL=INFO     # Normal
export LOG_LEVEL=WARNING  # Warnings and errors only
```

## Format

```
2026-05-04 10:23:15,045 [INFO] main [N/A]: Prediction request received — entity_id=4532015112830366
```

Includes: timestamp, level, logger name, request_id (from context if set), message.

## Docker

All logs go to stdout. View with:
```bash
docker logs -f modelserve_api
docker logs modelserve_api --tail 100
```

## Key Logs

**Startup**: "ModelServe starting up..." through "ready to serve predictions"

**Predictions**: request received → feature lookup → result with latency

**Training**: data load → class distribution → training progress → model registration

**Errors**: logged at ERROR level with full exception traceback via `logger.exception()`

**Features from Feast**: only logged at DEBUG level to avoid logging user data

## Request ID Tracking

Optional - for distributed tracing:
```python
from app.logger import set_request_id, clear_request_id

set_request_id("request-uuid-123")    # Includes in all logs from this context
clear_request_id()                     # Clear when done
```

Currently just logged as [N/A] since we don't set request_id from incoming requests yet.
