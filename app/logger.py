import logging
import os
import sys
from contextvars import ContextVar

request_id_context: ContextVar[str] = ContextVar("request_id", default="")


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        request_id = request_id_context.get()
        record.request_id = request_id if request_id else "N/A"
        return True


def configure_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Validate log level
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        log_level = "INFO"

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Remove any existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level))

    # Create formatter with request_id support
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    # Add request_id filter to handler
    handler.addFilter(RequestIdFilter())

    # Add handler to root logger
    root_logger.addHandler(handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        configure_logging()
    return logger


def set_request_id(request_id: str):
    request_id_context.set(request_id)


def clear_request_id():
    request_id_context.set("")
