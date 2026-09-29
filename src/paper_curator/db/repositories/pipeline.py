"""Persistence for pipeline execution state.

Runs, checkpoints, per-document progress and dead-letter records. None of this
is derived data: if these rows are wrong the pipeline either repeats work or
silently skips it, and neither failure announces itself.

The checkpoint deserves particular care. :func:`advance_checkpoint` is called
**after** a batch is fully processed, never before. A crash mid-batch then
causes that batch to be re-fetched, which is harmless because ingestion is
idempotent -- whereas advancing early would skip those papers permanently, and
nothing would ever notice.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from paper_curator.core.logging import get_logger
from paper_curator.db.enums import ProcessingStage, RunStatus, RunType
from paper_curator.db.models import (
    DocumentProcessingStatus,
    FailedDocument,
    IngestionRun,
    PipelineCheckpoint,
)

logger = get_logger(__name__)


def checkpoint_key_for(category: str) -> str:
    """Build the checkpoint key for one arXiv category stream."""
    return f"arxiv:{category}"


async def start_run(
    session: AsyncSession,
    *,
    run_type: RunType,
    triggered_by: str,
    config: dict[str, Any],
) -> IngestionRun:
    """Open a run record.

    ``config`` captures the exact parameters used -- categories, window, limits.
    Without it, an odd result months later cannot be explained, because the
    configuration that produced it is gone.
    """
    run = IngestionRun(
        run_type=run_type,
        status=RunStatus.RUNNING,
        started_at=dt.datetime.now(dt.UTC),
        triggered_by=triggered_by,
        config_snapshot=config,
    )
    session.add(run)
    await session.flush()
    logger.info("run_started", run_id=run.id, run_type=run_type.value, config=config)
    return run


async def finish_run(
    session: AsyncSession,
    run: IngestionRun,
    *,
    error: str | None = None,
) -> IngestionRun:
    """Close a run, deriving its final status from the counters.

    A run with failures is ``PARTIAL`` rather than ``FAILED``. At any scale some
    PDFs are malformed and some are pure scanned images; treating those as a
    whole-run failure would make the success signal meaningless and train
    everyone to ignore it.
    """
    run.finished_at = dt.datetime.now(dt.UTC)

    if error is not None:
        run.status = RunStatus.FAILED
        run.error_message = error
    elif run.papers_failed > 0:
        run.status = RunStatus.PARTIAL
    else:
        run.status = RunStatus.SUCCEEDED

    await session.flush()
    logger.info(
        "run_finished",
        run_id=run.id,
        status=run.status.value,
        fetched=run.papers_fetched,
        accepted=run.papers_accepted,
        skipped=run.papers_skipped,
        updated=run.papers_updated,
        failed=run.papers_failed,
        chunks_indexed=run.chunks_indexed,
    )
    return run


async def get_checkpoint(session: AsyncSession, key: str) -> PipelineCheckpoint | None:
    """Fetch a stream's resume point, or None when it has never run."""
    result = await session.execute(
        select(PipelineCheckpoint).where(PipelineCheckpoint.checkpoint_key == key)
    )
    return result.scalar_one_or_none()


async def advance_checkpoint(
    session: AsyncSession,
    *,
    key: str,
    updated_at_source: dt.datetime,
    last_arxiv_id: str | None = None,
    run_id: int | None = None,
) -> PipelineCheckpoint:
    """Move a stream's high-water mark forward.

    Refuses to move backwards. A late-arriving batch, or a backfill running
    alongside an incremental sync, would otherwise rewind the mark and cause
    everything since to be re-fetched.
    """
    checkpoint = await get_checkpoint(session, key)

    if checkpoint is None:
        checkpoint = PipelineCheckpoint(
            checkpoint_key=key,
            last_updated_at_source=updated_at_source,
            last_arxiv_id=last_arxiv_id,
            last_run_id=run_id,
        )
        session.add(checkpoint)
        await session.flush()
        logger.info("checkpoint_created", key=key, watermark=updated_at_source.isoformat())
        return checkpoint

    current = checkpoint.last_updated_at_source
    if current is not None and updated_at_source <= current:
        logger.debug(
            "checkpoint_not_advanced",
            key=key,
            current=current.isoformat(),
            proposed=updated_at_source.isoformat(),
        )
        return checkpoint

    checkpoint.last_updated_at_source = updated_at_source
    checkpoint.last_arxiv_id = last_arxiv_id
    checkpoint.last_run_id = run_id
    await session.flush()
    logger.info("checkpoint_advanced", key=key, watermark=updated_at_source.isoformat())
    return checkpoint


async def get_or_create_status(session: AsyncSession, paper_id: int) -> DocumentProcessingStatus:
    """Fetch a paper's processing record, creating it on first sight."""
    result = await session.execute(
        select(DocumentProcessingStatus).where(DocumentProcessingStatus.paper_id == paper_id)
    )
    status = result.scalar_one_or_none()
    if status is not None:
        return status

    status = DocumentProcessingStatus(paper_id=paper_id, stage=ProcessingStage.PENDING)
    session.add(status)
    await session.flush()
    return status


async def advance_stage(
    session: AsyncSession, status: DocumentProcessingStatus, stage: ProcessingStage
) -> None:
    """Record that a document reached a later stage.

    ``stage`` is the last *completed* step, so a crashed run resumes each
    document from where it actually got to rather than restarting the batch.
    """
    status.stage = stage
    status.last_attempt_at = dt.datetime.now(dt.UTC)
    if stage is ProcessingStage.INDEXED:
        # A clean finish clears the error from any earlier attempt, so a stale
        # message cannot be mistaken for a current problem.
        status.last_error = None
        status.last_error_type = None
        status.next_retry_at = None
    await session.flush()


async def record_attempt_failure(
    session: AsyncSession,
    status: DocumentProcessingStatus,
    *,
    error: Exception,
    backoff_seconds: float,
) -> bool:
    """Record a failed attempt and schedule the next one.

    Returns True when the retry budget is now exhausted, meaning the caller
    should write a dead-letter record.

    Backoff is stored as a timestamp rather than enforced by sleeping: the
    document is simply not eligible until that time passes, so the scheduler
    stays stateless and a pending retry survives a process restart.
    """
    status.attempts += 1
    status.last_attempt_at = dt.datetime.now(dt.UTC)
    status.last_error = str(error)[:2000]
    status.last_error_type = type(error).__name__

    if status.retries_exhausted:
        status.stage = ProcessingStage.FAILED
        status.next_retry_at = None
    else:
        status.next_retry_at = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=backoff_seconds)

    await session.flush()
    return status.retries_exhausted


async def record_dead_letter(
    session: AsyncSession,
    *,
    arxiv_id: str,
    version: int | None,
    stage: ProcessingStage,
    error: Exception,
    attempts: int,
    paper_id: int | None = None,
    run_id: int | None = None,
    payload: dict[str, Any] | None = None,
) -> FailedDocument:
    """Write a dead-letter record for a document that exhausted its retries.

    ``arxiv_id`` is stored directly rather than only referenced through
    ``paper_id``: the failures most worth studying are those where the paper row
    may never have been written, and a record that vanishes with its paper
    cannot be analysed.
    """
    failure = FailedDocument(
        paper_id=paper_id,
        arxiv_id=arxiv_id,
        version=version,
        stage=stage,
        error_type=type(error).__name__,
        error_message=str(error)[:4000],
        attempts=attempts,
        run_id=run_id,
        payload=payload or {},
    )
    session.add(failure)
    await session.flush()
    logger.warning(
        "document_dead_lettered",
        arxiv_id=arxiv_id,
        stage=stage.value,
        error_type=type(error).__name__,
        attempts=attempts,
    )
    return failure
