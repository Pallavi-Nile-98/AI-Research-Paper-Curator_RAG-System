"""Errors raised while downloading and extracting PDFs.

All inherit from :class:`~paper_curator.core.exceptions.IngestionError`, because
they describe one document failing rather than a service being down. That
distinction drives behaviour: a failed document is recorded against its paper
and retried within bounds, while the run continues. Some PDFs are malformed and
some are pure scanned images; treating either as a run-level failure would make
the success signal meaningless.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from paper_curator.core.exceptions import IngestionError


class PdfDownloadError(IngestionError):
    """A PDF could not be retrieved."""

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        status_code: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(context or {})
        if url is not None:
            merged["url"] = url
        if status_code is not None:
            merged["status_code"] = status_code
        super().__init__(message, context=merged)
        self.url = url
        self.status_code = status_code

    @property
    def is_retryable(self) -> bool:
        """True when a later attempt could plausibly succeed.

        A 404 means the PDF is not there and will not be there next minute. A
        timeout or a 503 is worth another try. No status code at all means the
        failure was below HTTP.
        """
        if self.status_code is None:
            return True
        return self.status_code == 429 or self.status_code >= 500


class PdfTooLargeError(PdfDownloadError):
    """The download exceeded the configured size limit.

    Separate from a generic download failure because it is never retryable and
    is a deliberate refusal rather than a fault: the file really is that big.
    """


class UnsafeUrlError(PdfDownloadError):
    """A URL was refused before any request was made.

    Raised when a PDF URL points somewhere outside the allowed hosts. PDF URLs
    arrive from the arXiv feed, which is third-party input, so a spoofed or
    compromised feed could otherwise aim the downloader at an internal address.
    """


class PdfExtractionError(IngestionError):
    """A PDF was retrieved but its content could not be read.

    Covers encrypted files, structurally corrupt documents, and anything the
    parser rejects outright. Distinct from *poor quality* extraction, which
    succeeds and is handled by the quality assessment rather than by an
    exception.
    """


class OcrUnavailableError(IngestionError):
    """OCR was required but the Tesseract binary is not installed.

    Deliberately not an extraction failure. The document may be perfectly
    processable elsewhere; this environment simply cannot do it, and the
    distinction matters when triaging a batch of failures.
    """
