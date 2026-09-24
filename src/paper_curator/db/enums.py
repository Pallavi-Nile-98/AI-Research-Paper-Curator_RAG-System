"""Enumerations used by the persistence layer.

These are stored as ``VARCHAR`` with a ``CHECK`` constraint rather than as native
PostgreSQL ``ENUM`` types. Native enums are tidier in the database but adding a
value to one requires ``ALTER TYPE``, which cannot run inside a transaction
block in older PostgreSQL versions and complicates rollback. Given that
``ProcessingStage`` will almost certainly gain values as the pipeline grows, the
string representation is the pragmatic choice: values remain validated, and
adding one is an ordinary migration.

All enums subclass :class:`~enum.StrEnum` so a member compares equal to its own
string value, which keeps log output and JSON serialisation readable.
"""

from __future__ import annotations

from enum import StrEnum


class RunType(StrEnum):
    """How an ingestion run was initiated."""

    INCREMENTAL = "incremental"
    """Routine sync: fetch everything newer than the stored checkpoint."""

    BACKFILL = "backfill"
    """Bounded historical fetch over an explicit date window."""

    MANUAL = "manual"
    """Ad-hoc run over an explicit list of arXiv identifiers."""


class RunStatus(StrEnum):
    """Terminal or in-flight state of an ingestion run."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    PARTIAL = "partial"
    """The run completed, but some documents failed.

    Distinguished from FAILED because it is the *normal* outcome at scale: some
    PDFs are malformed and some are pure scanned images. Treating those as a
    whole-run failure would make the success signal meaningless.
    """


class ProcessingStage(StrEnum):
    """Where a single document has reached in the pipeline.

    Ordered from earliest to latest. Each value is a *completed* state except
    ``PENDING`` and ``FAILED``, so a document at ``EXTRACTED`` has text and is
    waiting to be chunked.
    """

    PENDING = "pending"
    DOWNLOADED = "downloaded"
    EXTRACTED = "extracted"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    """Terminal success: chunks are searchable in OpenSearch."""

    FAILED = "failed"
    """Terminal failure: retries exhausted. See the failed_documents table."""

    SKIPPED = "skipped"
    """Deliberately not processed — for example, no PDF is available."""


class ExtractionMethod(StrEnum):
    """Which extractor produced a document's text.

    Recorded per document because it is a quality signal worth analysing: a
    corpus where OCR fires often indicates either many scanned papers or a
    miscalibrated quality threshold.
    """

    PYMUPDF = "pymupdf"
    """Primary path: the PDF contained an extractable text layer."""

    OCR = "ocr"
    """Fallback: pages were rendered to images and read with Tesseract."""

    NONE = "none"
    """Extraction produced nothing usable."""


class RetrievalMode(StrEnum):
    """Which retrieval strategy served a query.

    Persisted with every query so measured latency and user feedback can be
    attributed to a specific configuration rather than averaged across all of
    them. See ADR-0004.
    """

    KEYWORD = "keyword"
    """BM25 lexical scoring only."""

    VECTOR = "vector"
    """Dense embedding similarity only."""

    HYBRID = "hybrid"
    """Both, fused by rank."""


class FeedbackRating(StrEnum):
    """A user's judgement of an answer.

    Deliberately binary. A five-point scale invites inconsistent interpretation
    between users and yields little more signal at this volume.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
