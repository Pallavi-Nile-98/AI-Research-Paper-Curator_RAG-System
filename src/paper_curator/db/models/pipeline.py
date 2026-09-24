"""Models that track pipeline execution, progress and failure.

Nothing here is derived data — this is the state that makes ingestion resumable
and idempotent. If these tables are wrong, the pipeline either does work twice
or silently skips it.

Three concerns are separated deliberately:

* :class:`IngestionRun` — what happened during one execution.
* :class:`PipelineCheckpoint` — where to resume from next time.
* :class:`DocumentProcessingStatus` / :class:`FailedDocument` — per-document
  progress, and the dead-letter record when retries are exhausted.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from paper_curator.db.base import Base, TimestampMixin
from paper_curator.db.enums import ProcessingStage, RunStatus, RunType

if TYPE_CHECKING:
    from paper_curator.db.models.paper import Paper


class IngestionRun(Base, TimestampMixin):
    """One execution of the ingestion pipeline.

    The counters exist so a run can be judged without reading logs. A run that
    accepted 3 papers and failed 97 is technically ``PARTIAL`` and practically a
    problem; the numbers make that visible immediately.
    """

    __tablename__ = "ingestion_runs"
    __table_args__ = (
        Index("ix_ingestion_runs_started_at", "started_at"),
        Index("ix_ingestion_runs_status", "status"),
        CheckConstraint(
            "finished_at IS NULL OR finished_at >= started_at",
            name="finished_after_started",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)

    run_type: Mapped[RunType] = mapped_column(
        Enum(RunType, native_enum=False, length=16), nullable=False
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, native_enum=False, length=16),
        nullable=False,
        default=RunStatus.RUNNING,
    )

    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    # Counters. Named to match the four outcomes the pipeline logs, so a run
    # summary and the log stream describe the same thing.
    papers_fetched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    papers_accepted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    papers_skipped: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, doc="Already present at this version"
    )
    papers_updated: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, doc="A newer version replaced an older one"
    )
    papers_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_indexed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    triggered_by: Mapped[str] = mapped_column(
        String(32), nullable=False, default="cli", doc="cli | airflow | api"
    )
    # The exact parameters this run used -- categories, date window, limits.
    # Without it, an unexpected result months later cannot be explained, because
    # the configuration that produced it is gone.
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<IngestionRun {self.id} {self.run_type} {self.status}>"


class PipelineCheckpoint(Base, TimestampMixin):
    """Resume point for incremental synchronisation.

    One row per logical stream — typically one per arXiv category — holding a
    high-water mark. The next incremental run fetches only what is newer,
    turning a daily job into a small delta rather than a full re-scan.

    The watermark is the source's *updated* timestamp, not its published
    timestamp. A 2019 paper revised yesterday must be picked up; watermarking on
    publication date would miss every revision to older work.

    A deliberate consequence: the watermark is advanced only after a batch is
    fully processed, so a crash mid-batch causes that batch to be re-fetched.
    Re-fetching is harmless because ingestion is idempotent, whereas advancing
    early would silently skip papers forever.
    """

    __tablename__ = "pipeline_checkpoints"
    __table_args__ = (
        UniqueConstraint("checkpoint_key", name="uq_pipeline_checkpoints_checkpoint_key"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    checkpoint_key: Mapped[str] = mapped_column(
        String(128), nullable=False, doc="Stream identifier, e.g. 'arxiv:cs.CL'"
    )
    last_updated_at_source: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), doc="High-water mark; NULL means never run"
    )
    last_arxiv_id: Mapped[str | None] = mapped_column(
        String(32), doc="Tie-breaker when several papers share a timestamp"
    )
    last_run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("ingestion_runs.id", ondelete="SET NULL")
    )

    def __repr__(self) -> str:
        return f"<Checkpoint {self.checkpoint_key} @ {self.last_updated_at_source}>"


class DocumentProcessingStatus(Base, TimestampMixin):
    """Per-document progress through the pipeline, with bounded retries.

    One row per paper version. ``stage`` records the last *completed* step, so a
    crashed run resumes from where each document actually got to rather than
    restarting the whole batch.

    ``next_retry_at`` implements backoff as data rather than as a sleeping
    process: a failed document is simply not eligible until that time passes, so
    the scheduler stays stateless and a retry survives a restart.
    """

    __tablename__ = "document_processing_status"
    __table_args__ = (
        UniqueConstraint("paper_id", name="uq_document_processing_status_paper_id"),
        # The scheduler's main query: what is ready to work on right now?
        Index("ix_document_processing_status_stage_retry", "stage", "next_retry_at"),
        CheckConstraint("attempts >= 0", name="attempts_non_negative"),
        CheckConstraint("max_attempts >= 1", name="max_attempts_positive"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    paper_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )

    stage: Mapped[ProcessingStage] = mapped_column(
        Enum(ProcessingStage, native_enum=False, length=16),
        nullable=False,
        default=ProcessingStage.PENDING,
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    last_attempt_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    next_retry_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), doc="Not eligible for processing before this time"
    )
    last_error: Mapped[str | None] = mapped_column(Text)
    last_error_type: Mapped[str | None] = mapped_column(
        String(128), doc="Exception class name, for grouping failures by cause"
    )

    paper: Mapped[Paper] = relationship(back_populates="processing_status")

    @property
    def retries_exhausted(self) -> bool:
        """True when this document has used its full retry budget."""
        return self.attempts >= self.max_attempts

    def __repr__(self) -> str:
        return f"<Status paper={self.paper_id} {self.stage} attempts={self.attempts}>"


class FailedDocument(Base, TimestampMixin):
    """Dead-letter record for a document that exhausted its retries.

    ``arxiv_id`` is denormalised rather than only referenced through
    ``paper_id``. A failure record whose meaning depends on a row that may be
    deleted is not much of a failure record, and the failures worth studying are
    precisely the ones where the paper row may never have been written.

    ``payload`` holds whatever context helps diagnosis — the URL attempted, the
    HTTP status, how many characters extraction produced. Deliberately
    unstructured, because the useful field differs per failure mode.
    """

    __tablename__ = "failed_documents"
    __table_args__ = (
        Index("ix_failed_documents_arxiv_id", "arxiv_id"),
        Index("ix_failed_documents_stage", "stage"),
        Index("ix_failed_documents_resolved_at", "resolved_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)

    paper_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("papers.id", ondelete="SET NULL")
    )
    arxiv_id: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[int | None] = mapped_column(Integer)

    stage: Mapped[ProcessingStage] = mapped_column(
        Enum(ProcessingStage, native_enum=False, length=16), nullable=False
    )
    error_type: Mapped[str] = mapped_column(String(128), nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("ingestion_runs.id", ondelete="SET NULL")
    )
    # Set when the document is reprocessed successfully or triaged as
    # permanently unprocessable. Retained rather than deleted so the failure
    # rate over time stays measurable.
    resolved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_note: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<FailedDocument {self.arxiv_id} {self.stage} {self.error_type}>"
