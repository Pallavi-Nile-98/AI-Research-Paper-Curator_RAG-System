"""Structured logging built on ``structlog``.

Why structured logs rather than formatted strings
-------------------------------------------------
A line like ``"indexed 42 chunks for 2401.12345 in 1.3s"`` is readable but not
queryable. The same event emitted as ``{"event": "chunks_indexed", "chunk_count": 42,
"arxiv_id": "2401.12345", "duration_ms": 1300}`` can be filtered, aggregated and
alerted on. Once this system is answering questions in production, "what is the p95
retrieval latency for hybrid mode?" has to be answerable from logs alone.

Two output formats
------------------
* ``console`` -- colourised, human-readable. Local development.
* ``json``    -- one JSON object per line. CI, Docker, and CloudWatch Logs, which
  parses JSON automatically and makes fields queryable in Logs Insights.

Context propagation
-------------------
``bind_request_context()`` writes into a :mod:`contextvars` store that structlog merges
into every subsequent log record on the same task. A request ID bound once in API
middleware therefore appears on the retrieval log, the re-ranking log and the generation
log without being threaded through a dozen function signatures. This works correctly
under asyncio, where many requests share one thread.
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from paper_curator.core.config import LogFormat, LogLevel

_REQUEST_ID_KEY = "request_id"


def _drop_color_message_key(_logger: object, _method_name: str, event_dict: EventDict) -> EventDict:
    """Remove uvicorn's duplicated ``color_message`` field."""
    event_dict.pop("color_message", None)
    return event_dict


def _shared_processors() -> list[Processor]:
    """Processors applied to both structlog and stdlib log records."""
    return [
        # Merge anything bound via bind_request_context() into this record.
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _drop_color_message_key,
    ]


def configure_logging(
    *,
    level: LogLevel = "INFO",
    log_format: LogFormat = "console",
) -> None:
    """Configure structlog and route the standard library through it.

    Routing stdlib logging through the same pipeline matters: uvicorn, SQLAlchemy,
    Alembic and ``opensearch-py`` all use ``logging``. Without this, half the output
    would be JSON and half would be plain text, and neither would carry the request ID.

    Safe to call more than once; later calls replace the previous configuration.
    """
    shared = _shared_processors()

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            *shared,
            # Hand off to the stdlib formatter below rather than rendering here.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # foreign_pre_chain runs on records that came from stdlib logging, not structlog.
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Replace existing handlers so repeated calls do not duplicate every line.
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # These libraries are extremely chatty at INFO and drown out application events.
    for noisy in ("urllib3", "opensearch", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, conventionally ``get_logger(__name__)``."""
    return structlog.stdlib.get_logger(name)


def bind_request_context(**values: Any) -> None:
    """Bind key/value pairs onto every subsequent log record in this async context."""
    structlog.contextvars.bind_contextvars(**values)


def new_request_id() -> str:
    """Generate a correlation ID for a single inbound request."""
    return uuid.uuid4().hex


def clear_request_context() -> None:
    """Clear all context-local bindings.

    Call at the end of a request so IDs never leak between requests sharing a worker.
    """
    structlog.contextvars.clear_contextvars()


__all__ = [
    "bind_request_context",
    "clear_request_context",
    "configure_logging",
    "get_logger",
    "new_request_id",
]
