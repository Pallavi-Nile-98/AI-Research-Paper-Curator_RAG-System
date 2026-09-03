"""Cross-cutting concerns shared by every other package: configuration, logging, errors.

Nothing in ``core`` may import from ``ingestion``, ``retrieval``, ``generation`` or
``api``. Dependencies point inwards only, which keeps this package trivially testable
and prevents circular imports as the codebase grows.
"""

from __future__ import annotations

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.exceptions import (
    ConfigurationError,
    ExternalServiceError,
    GenerationError,
    IngestionError,
    PaperCuratorError,
    RetrievalError,
)
from paper_curator.core.logging import bind_request_context, configure_logging, get_logger

__all__ = [
    "ConfigurationError",
    "ExternalServiceError",
    "GenerationError",
    "IngestionError",
    "PaperCuratorError",
    "RetrievalError",
    "Settings",
    "bind_request_context",
    "configure_logging",
    "get_logger",
    "get_settings",
]
