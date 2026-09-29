"""The ingestion pipeline: arXiv to searchable index.

Every stage built in earlier batches, wired into one flow:

    fetch metadata -> store -> download PDF -> extract text (OCR if needed)
    -> chunk -> embed -> index -> mark indexed

Three decisions shape how this is written.

**One transaction per paper.** A batch-wide transaction would mean one
malformed PDF rolling back forty papers that processed cleanly. Per-paper
commits make a partial run genuinely partial: the work that succeeded is
durable, and only the failures are retried.

**Failures are recorded, not raised.** A run that aborts on the first bad PDF
never reaches the good papers behind it. Some documents are malformed and some
are pure scanned images; that is ordinary, so each failure is written against
its document and the run continues.

**The checkpoint advances last.** Only after a paper is fully indexed. A crash
mid-run therefore re-fetches some papers, which is harmless because ingestion
is idempotent -- whereas advancing early would skip them permanently and
nothing would ever notice.

This module holds the orchestration logic so that the CLI and the eventual
Airflow DAG are both thin callers, and neither becomes a place business logic
hides.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

from opensearchpy import AsyncOpenSearch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.logging import bind_request_context, get_logger, new_request_id
from paper_curator.core.retry import compute_backoff
from paper_curator.db import repositories as repo
from paper_curator.db.enums import ProcessingStage, RunType
from paper_curator.db.models import IngestionRun, Paper
from paper_curator.db.session import session_scope
from paper_curator.embedding import EmbeddingProvider, get_embedding_provider
from paper_curator.ingestion.arxiv_client import ArxivClient, ArxivPaper, ArxivQuery
from paper_curator.ingestion.chunking import StructureAwareChunker, build_chunk_document
from paper_curator.ingestion.extraction import PdfExtractionService
from paper_curator.search import ChunkIndexer, IndexManager, create_client

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PaperOutcome:
    """What happened to one paper."""

    arxiv_id: str
    version: int
    outcome: str
    chunks_indexed: int = 0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class RunSummary:
    """Aggregate result of one ingestion run."""

    run_id: int | None = None
    fetched: int = 0
    accepted: int = 0
    skipped: int = 0
    updated: int = 0
    failed: int = 0
    chunks_indexed: int = 0
    duration_seconds: float = 0.0
    outcomes: list[PaperOutcome] = field(default_factory=list)

    def describe(self) -> str:
        """One-line summary for a terminal or a log."""
        return (
            f"fetched={self.fetched} accepted={self.accepted} updated={self.updated} "
            f"skipped={self.skipped} failed={self.failed} "
            f"chunks={self.chunks_indexed} in {self.duration_seconds:.1f}s"
        )


class IngestionPipeline:
    """Runs papers from the arXiv API through to a searchable index.

    Dependencies are injected so each can be replaced in tests -- a fake
    embedding provider instead of a 2 GB model, a mocked HTTP transport instead
    of the live API.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        arxiv_client: ArxivClient | None = None,
        extraction: PdfExtractionService | None = None,
        chunker: StructureAwareChunker | None = None,
        embedder: EmbeddingProvider | None = None,
        indexer: ChunkIndexer | None = None,
        index_manager: IndexManager | None = None,
    ) -> None:
        self._settings = settings or get_settings()

        self._owns_clients = arxiv_client is None
        self._arxiv = arxiv_client or ArxivClient(self._settings)
        self._extraction = extraction or PdfExtractionService(self._settings)
        self._chunker = chunker or StructureAwareChunker(self._settings)
        self._embedder = embedder or get_embedding_provider(self._settings)

        self._search_client: AsyncOpenSearch | None = None
        if indexer is None or index_manager is None:
            self._search_client = create_client(self._settings)
            self._indexer = indexer or ChunkIndexer(self._search_client, self._settings)
            self._index_manager = index_manager or IndexManager(self._search_client, self._settings)
        else:
            self._indexer = indexer
            self._index_manager = index_manager

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release connections this pipeline created."""
        if self._owns_clients:
            await self._arxiv.aclose()
            await self._extraction.aclose()
        if self._search_client is not None:
            await self._search_client.close()

    # ------------------------------------------------------------------
    # Per-paper processing
    # ------------------------------------------------------------------

    async def _process_paper(self, fetched: ArxivPaper, run: IngestionRun | None) -> PaperOutcome:
        """Take one paper from metadata to indexed, across two transactions."""
        # --- Transaction 1: metadata, committed on its own -------------------
        #
        # Deliberately separate from content processing. Sharing one
        # transaction meant a failed PDF download rolled back the paper row
        # with it, leaving nothing to record retries against -- so a document
        # was dead-lettered on its first attempt and max_attempts was
        # decorative. Committing metadata first makes retry state durable.
        try:
            async with session_scope() as session:
                stored = await repo.upsert_paper(session, fetched)
                status = await repo.get_or_create_status(session, stored.paper.id)
                paper_id = stored.paper.id
                upsert_outcome = stored.outcome.value
                # A new or updated paper always needs processing. For one
                # already stored, the answer comes from its processing state,
                # not from the row's existence -- otherwise anything stored but
                # unfinished is skipped forever and the retry columns are
                # write-only.
                is_new = stored.needs_processing
                needs_work = is_new or status.needs_work(dt.datetime.now(dt.UTC))
                attempts_so_far = status.attempts
        except Exception as exc:
            await self._record_failure(fetched, exc, run)
            logger.warning(
                "paper_metadata_failed",
                arxiv_id=fetched.arxiv_id,
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            return PaperOutcome(
                arxiv_id=fetched.arxiv_id,
                version=fetched.version,
                outcome="failed",
                error=f"{type(exc).__name__}: {exc}"[:500],
            )

        if not needs_work:
            return PaperOutcome(
                arxiv_id=fetched.arxiv_id,
                version=fetched.version,
                outcome=upsert_outcome,
            )

        if not is_new:
            logger.info(
                "retrying_unfinished_paper",
                arxiv_id=fetched.arxiv_id,
                attempts=attempts_so_far,
            )

        # --- Transaction 2: content ------------------------------------------
        try:
            async with session_scope() as session:
                indexed = await self._process_content(session, fetched, paper_id)
                status = await repo.get_or_create_status(session, paper_id)
                await repo.advance_stage(session, status, ProcessingStage.INDEXED)

                return PaperOutcome(
                    arxiv_id=fetched.arxiv_id,
                    version=fetched.version,
                    outcome=upsert_outcome,
                    chunks_indexed=indexed,
                )
        except Exception as exc:
            # Recorded in a FRESH session, deliberately.
            #
            # If the failure came from the database -- a PDF containing a NUL
            # byte, say -- the session above is already rolled back and every
            # further statement on it raises PendingRollbackError. An earlier
            # version recorded the failure on that same session, so a single bad
            # paper turned into an aborted run that discarded the papers already
            # processed. The recovery path must not depend on the thing that
            # just broke.
            await self._record_failure(fetched, exc, run)
            logger.warning(
                "paper_processing_failed",
                arxiv_id=fetched.arxiv_id,
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            return PaperOutcome(
                arxiv_id=fetched.arxiv_id,
                version=fetched.version,
                outcome="failed",
                error=f"{type(exc).__name__}: {exc}"[:500],
            )

    async def _record_failure(
        self, fetched: ArxivPaper, exc: Exception, run: IngestionRun | None
    ) -> None:
        """Record a failed attempt, and dead-letter it once retries run out.

        Swallows its own errors. Failing to record a failure must not replace
        the original exception with a less informative one, and must never take
        down a run that still has papers to process.
        """
        try:
            async with session_scope() as session:
                paper = (
                    await session.execute(
                        select(Paper).where(
                            Paper.arxiv_id == fetched.arxiv_id,
                            Paper.version == fetched.version,
                        )
                    )
                ).scalar_one_or_none()

                if paper is None:
                    # The paper row itself never landed, so there is no status
                    # to update. The dead-letter record carries arxiv_id
                    # directly for exactly this case.
                    await repo.record_dead_letter(
                        session,
                        arxiv_id=fetched.arxiv_id,
                        version=fetched.version,
                        stage=ProcessingStage.PENDING,
                        error=exc,
                        attempts=1,
                        run_id=run.id if run else None,
                        payload={"pdf_url": fetched.pdf_url},
                    )
                    return

                status = await repo.get_or_create_status(session, paper.id)
                exhausted = await repo.record_attempt_failure(
                    session,
                    status,
                    error=exc,
                    backoff_seconds=compute_backoff(
                        status.attempts, base_delay=60.0, max_delay=3600.0, jitter=False
                    ),
                )
                if exhausted:
                    await repo.record_dead_letter(
                        session,
                        arxiv_id=fetched.arxiv_id,
                        version=fetched.version,
                        stage=status.stage,
                        error=exc,
                        attempts=status.attempts,
                        paper_id=paper.id,
                        run_id=run.id if run else None,
                        payload={"pdf_url": fetched.pdf_url},
                    )
        except Exception:
            logger.exception(
                "failure_recording_failed",
                arxiv_id=fetched.arxiv_id,
                original_error=type(exc).__name__,
            )

    async def _process_content(
        self, session: AsyncSession, fetched: ArxivPaper, paper_id: int
    ) -> int:
        """Download, extract, chunk, embed and index one paper's PDF."""
        extraction = await self._extraction.extract_from_url(fetched.pdf_url)

        await repo.save_document_text(
            session,
            paper_id=paper_id,
            text=extraction.text,
            method=extraction.method,
            page_count=extraction.page_count,
            quality_score=extraction.quality.score,
        )

        if not extraction.is_usable:
            # Not an error: a scanned paper with no OCR available has nothing to
            # index. The text row records why, and the paper is not retried
            # indefinitely for a condition retrying cannot change.
            logger.warning(
                "paper_has_no_usable_text",
                arxiv_id=fetched.arxiv_id,
                method=extraction.method.value,
                reason=extraction.ocr_skipped_reason,
            )
            return 0

        chunks = self._chunker.chunk(extraction.text)
        if not chunks:
            return 0

        await repo.replace_chunks(
            session,
            paper_id=paper_id,
            arxiv_id=fetched.arxiv_id,
            version=fetched.version,
            chunks=chunks,
            embedding_model=self._embedder.model_name,
        )

        # Embedding is CPU-bound; keep it off the event loop.
        vectors = await asyncio.to_thread(
            self._embedder.embed_documents, [chunk.text for chunk in chunks]
        )

        now = dt.datetime.now(dt.UTC).isoformat()
        documents = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            document = build_chunk_document(
                chunk,
                arxiv_id=fetched.arxiv_id,
                version=fetched.version,
                title=fetched.title,
                authors=list(fetched.authors),
                abstract=fetched.abstract,
                published_at=fetched.published_at,
                updated_at=fetched.updated_at,
                primary_category=fetched.primary_category,
                categories=list(fetched.categories),
                abs_url=fetched.abs_url,
                pdf_url=fetched.pdf_url,
            )
            document["embedding"] = vector
            document["embedding_model"] = self._embedder.model_name
            document["indexed_at"] = now
            documents.append(document)

        result = await self._indexer.index_documents(documents)

        # Only the ids OpenSearch actually accepted. The rest keep
        # indexed_at NULL and are reconciled by a later run.
        await repo.mark_chunks_indexed(session, result.indexed_ids)

        if not result.is_complete:
            logger.warning(
                "paper_partially_indexed",
                arxiv_id=fetched.arxiv_id,
                indexed=result.indexed_count,
                failed=result.failure_count,
            )
        return result.indexed_count

    # ------------------------------------------------------------------
    # Run orchestration
    # ------------------------------------------------------------------

    async def run(
        self,
        query: ArxivQuery,
        *,
        limit: int,
        run_type: RunType = RunType.INCREMENTAL,
        triggered_by: str = "cli",
        checkpoint_key: str | None = None,
    ) -> RunSummary:
        """Fetch papers matching ``query`` and take each through the pipeline."""
        bind_request_context(request_id=new_request_id())
        started = time.perf_counter()
        summary = RunSummary()

        await self._index_manager.ensure_ready()

        async with session_scope() as session:
            run = await repo.start_run(
                session,
                run_type=run_type,
                triggered_by=triggered_by,
                config={
                    "search_query": query.to_search_query(),
                    "limit": limit,
                    "checkpoint_key": checkpoint_key,
                    "embedding_model": self._embedder.model_name,
                },
            )
            run_id = run.id
        summary.run_id = run_id

        highest_watermark: dt.datetime | None = None
        last_arxiv_id: str | None = None
        error: str | None = None

        try:
            async for fetched in self._arxiv.iter_papers(query, limit=limit):
                summary.fetched += 1
                outcome = await self._process_paper(fetched, run)
                summary.outcomes.append(outcome)

                if outcome.error is not None:
                    summary.failed += 1
                elif outcome.outcome == "created":
                    summary.accepted += 1
                elif outcome.outcome == "updated":
                    summary.updated += 1
                else:
                    summary.skipped += 1

                summary.chunks_indexed += outcome.chunks_indexed

                # Track the watermark but do not persist it yet.
                if outcome.succeeded and (
                    highest_watermark is None or fetched.updated_at > highest_watermark
                ):
                    highest_watermark = fetched.updated_at
                    last_arxiv_id = fetched.arxiv_id
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("run_aborted", run_id=run_id)
            raise
        finally:
            summary.duration_seconds = time.perf_counter() - started
            async with session_scope() as session:
                stored_run = await session.get(IngestionRun, run_id)
                if stored_run is not None:
                    stored_run.papers_fetched = summary.fetched
                    stored_run.papers_accepted = summary.accepted
                    stored_run.papers_skipped = summary.skipped
                    stored_run.papers_updated = summary.updated
                    stored_run.papers_failed = summary.failed
                    stored_run.chunks_indexed = summary.chunks_indexed
                    await repo.finish_run(session, stored_run, error=error)

                # Advance only now, and only past papers that actually
                # succeeded. A crash before this point re-fetches the batch,
                # which idempotency makes harmless.
                if checkpoint_key and highest_watermark is not None and error is None:
                    await repo.advance_checkpoint(
                        session,
                        key=checkpoint_key,
                        updated_at_source=highest_watermark,
                        last_arxiv_id=last_arxiv_id,
                        run_id=run_id,
                    )

        logger.info("run_summary", run_id=run_id, summary=summary.describe())
        return summary

    async def run_incremental(
        self, *, category: str, limit: int, triggered_by: str = "cli"
    ) -> RunSummary:
        """Fetch only what is newer than this category's stored checkpoint.

        Filters on the source's *updated* timestamp, not its publication date,
        so a 2019 paper revised yesterday is picked up. Watermarking on
        publication would miss every revision to older work.
        """
        key = repo.checkpoint_key_for(category)

        async with session_scope() as session:
            checkpoint = await repo.get_checkpoint(session, key)
            since = checkpoint.last_updated_at_source if checkpoint else None

        if since is None:
            logger.info("incremental_first_run", category=category)
        else:
            logger.info("incremental_resuming", category=category, since=since.isoformat())

        return await self.run(
            ArxivQuery(categories=[category], updated_after=since),
            limit=limit,
            run_type=RunType.INCREMENTAL,
            triggered_by=triggered_by,
            checkpoint_key=key,
        )

    async def run_backfill(
        self,
        *,
        category: str,
        since: dt.datetime,
        until: dt.datetime | None = None,
        limit: int,
        triggered_by: str = "cli",
    ) -> RunSummary:
        """Fetch a bounded historical window.

        Both a date range and a limit are required. An unbounded backfill over a
        broad category pages through tens of thousands of results at three
        seconds per request -- hours of wall-clock time, mostly unattended.

        Does not touch the checkpoint: a backfill fills a gap behind the
        watermark, and moving it would cause everything since to be re-fetched.
        """
        return await self.run(
            ArxivQuery(categories=[category], submitted_after=since, submitted_before=until),
            limit=limit,
            run_type=RunType.BACKFILL,
            triggered_by=triggered_by,
            checkpoint_key=None,
        )
