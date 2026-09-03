"""Application exception hierarchy.

Two rules make this hierarchy worth having:

1. **Every application error inherits from :class:`PaperCuratorError`.** API middleware
   can therefore catch one base class, map it to an HTTP status, and let genuinely
   unexpected exceptions (``KeyError``, ``AttributeError``) surface as 500s rather than
   being silently swallowed by a bare ``except Exception``.
2. **Errors carry structured context, not interpolated strings.** ``context`` is a plain
   dict that goes straight into a structured log record, so failures stay queryable
   ("show me every OCR failure for arXiv ID 2401.*") instead of needing regex over
   free-form messages.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class PaperCuratorError(Exception):
    """Base class for every error raised deliberately by this application."""

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context) if context else {}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
        return f"{self.message} ({details})"


class ConfigurationError(PaperCuratorError):
    """Invalid, missing or mutually inconsistent configuration.

    Raised at startup rather than at first use, so a misconfigured deployment fails
    immediately and visibly instead of degrading under load.
    """


class ExternalServiceError(PaperCuratorError):
    """A call to a service outside this process failed.

    Covers arXiv, OpenSearch, Postgres, Ollama and Langfuse. The ``service`` attribute
    lets retry policies and dashboards distinguish a flaky search cluster from a
    genuinely unavailable model server.
    """

    def __init__(
        self,
        message: str,
        *,
        service: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context={**(context or {}), "service": service})
        self.service = service


class IngestionError(PaperCuratorError):
    """A paper could not be fetched, downloaded, extracted or chunked.

    Ingestion errors are expected in normal operation -- some PDFs are malformed, some
    are pure scanned images. They are recorded against the document and retried within
    bounds rather than aborting an entire pipeline run.
    """


class RetrievalError(PaperCuratorError):
    """Keyword search, vector search, fusion or re-ranking failed."""


class GenerationError(PaperCuratorError):
    """The LLM call failed, timed out, or returned unusable output."""


class EvaluationError(PaperCuratorError):
    """An evaluation dataset or experiment run is invalid."""


__all__ = [
    "ConfigurationError",
    "EvaluationError",
    "ExternalServiceError",
    "GenerationError",
    "IngestionError",
    "PaperCuratorError",
    "RetrievalError",
]
