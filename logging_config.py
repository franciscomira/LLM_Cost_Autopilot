"""
logging_config.py

Structured JSON logging + per-request correlation ID.

Usage in api.py:
    from logging_config import configure_logging, REQUEST_ID
    configure_logging()

Usage in any module:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("routed request", extra={"backend": "ollama"})
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        obj: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": REQUEST_ID.get(),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        # Attach any extra fields the caller passed via `extra={}`
        skip = logging.LogRecord.__dict__.keys() | {
            "message", "asctime", "args", "msg",
            "pathname", "filename", "module", "funcName", "lineno",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "exc_info", "exc_text", "stack_info",
            "levelno", "levelname", "name",
        }
        for k, v in record.__dict__.items():
            if k not in skip:
                obj[k] = v
        return json.dumps(obj, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Quiet noisy uvicorn access logs (we emit our own)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
