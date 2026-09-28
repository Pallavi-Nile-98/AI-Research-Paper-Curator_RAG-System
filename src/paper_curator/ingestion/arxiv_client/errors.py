"""Errors raised by the arXiv client.

Split into two hierarchies on purpose, because they demand different responses:

* :class:`ArxivApiError` is an :class:`~paper_curator.core.exceptions.ExternalServiceError` —
  the network or the service failed. Often transient, and the retry policy cares
  about the distinction between a timeout and a 400.
* :class:`ArxivParseError` is an :class:`~paper_curator.core.exceptions.IngestionError` —
  a response arrived but could not be understood. Retrying cannot fix it; the
  entry is recorded as failed and the run continues.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from paper_curator.core.exceptions import ExternalServiceError, IngestionError

SERVICE_NAME = "arxiv"


class ArxivApiError(ExternalServiceError):
    """A request to the arXiv API failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(context or {})
        if status_code is not None:
            merged["status_code"] = status_code
        super().__init__(message, service=SERVICE_NAME, context=merged)
        self.status_code = status_code

    @property
    def is_retryable(self) -> bool:
        """True when a later attempt could plausibly succeed.

        Retrying a 400 only repeats a rejected request. Retrying a 429 or a 5xx
        is the whole point of having a retry policy. A missing status code means
        the failure happened below HTTP — a timeout or a connection reset —
        which is the most retryable case of all.
        """
        if self.status_code is None:
            return True
        return self.status_code == 429 or self.status_code >= 500


class ArxivParseError(IngestionError):
    """An arXiv response could not be parsed into the expected structure."""
