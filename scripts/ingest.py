"""Run the ingestion pipeline from the command line.

    python scripts/ingest.py --category cs.CL --limit 5
    python scripts/ingest.py --backfill --since 2024-01-01 --until 2024-01-31 --limit 10
    python scripts/ingest.py --status

Deliberately thin. Every decision -- what is new, what to retry, when to
advance the checkpoint -- lives in
:class:`~paper_curator.ingestion.pipeline.IngestionPipeline`, so this file and
the eventual Airflow DAG are both callers rather than two places the same logic
has to be kept in step.

Requires PostgreSQL and OpenSearch to be running:

    docker-compose up -d

The first run downloads the embedding model (roughly 130 MB) and is slower than
later ones. arXiv is contacted at their requested three seconds between
requests, so fetching is paced rather than fast.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys

from sqlalchemy import func, select

from paper_curator.core.config import get_settings
from paper_curator.core.logging import configure_logging
from paper_curator.db.enums import ProcessingStage
from paper_curator.db.models import (
    Chunk,
    DocumentProcessingStatus,
    FailedDocument,
    IngestionRun,
    Paper,
    PipelineCheckpoint,
)
from paper_curator.db.session import dispose_engine, session_scope
from paper_curator.ingestion.pipeline import IngestionPipeline, RunSummary

LINE = "-" * 74


def _parse_date(value: str) -> dt.datetime:
    """Parse a YYYY-MM-DD argument into a UTC datetime."""
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=dt.UTC)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


def print_summary(summary: RunSummary) -> None:
    """Render a run summary."""
    print()
    print(LINE)
    print(f"Run #{summary.run_id} finished in {summary.duration_seconds:.1f}s")
    print(LINE)
    print(f"  fetched from arXiv : {summary.fetched}")
    print(f"  newly stored       : {summary.accepted}")
    print(f"  updated version    : {summary.updated}")
    print(f"  already had        : {summary.skipped}")
    print(f"  failed             : {summary.failed}")
    print(f"  chunks indexed     : {summary.chunks_indexed}")
    print(LINE)

    failures = [outcome for outcome in summary.outcomes if not outcome.succeeded]
    if failures:
        print("\nFailures:")
        for outcome in failures:
            print(f"  {outcome.arxiv_id}v{outcome.version}: {outcome.error}")


async def show_status() -> int:
    """Print what the pipeline currently holds, without changing anything."""
    async with session_scope() as session:
        papers = await session.scalar(select(func.count()).select_from(Paper))
        latest = await session.scalar(
            select(func.count()).select_from(Paper).where(Paper.is_latest.is_(True))
        )
        chunks = await session.scalar(select(func.count()).select_from(Chunk))
        unindexed = await session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.indexed_at.is_(None))
        )
        failed = await session.scalar(
            select(func.count())
            .select_from(FailedDocument)
            .where(FailedDocument.resolved_at.is_(None))
        )
        indexed_docs = await session.scalar(
            select(func.count())
            .select_from(DocumentProcessingStatus)
            .where(DocumentProcessingStatus.stage == ProcessingStage.INDEXED)
        )

        checkpoints = (await session.execute(select(PipelineCheckpoint))).scalars().all()
        runs = (
            (
                await session.execute(
                    select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(5)
                )
            )
            .scalars()
            .all()
        )

    print(LINE)
    print("Pipeline status")
    print(LINE)
    print(f"  papers stored          : {papers}  ({latest} at latest version)")
    print(f"  documents fully indexed: {indexed_docs}")
    print(f"  chunks                 : {chunks}")
    print(f"  chunks not yet indexed : {unindexed}")
    print(f"  unresolved failures    : {failed}")

    if unindexed:
        print("\n  Note: chunks with no index entry are reconciled by the next run.")

    print("\nCheckpoints:")
    if checkpoints:
        for checkpoint in checkpoints:
            mark = (
                checkpoint.last_updated_at_source.isoformat()
                if checkpoint.last_updated_at_source
                else "never run"
            )
            print(f"  {checkpoint.checkpoint_key:24} {mark}")
    else:
        print("  (none yet)")

    print("\nRecent runs:")
    if runs:
        for run in runs:
            print(
                f"  #{run.id:<4} {run.run_type.value:<12} {run.status.value:<10} "
                f"fetched={run.papers_fetched:<4} failed={run.papers_failed:<4} "
                f"chunks={run.chunks_indexed}"
            )
    else:
        print("  (none yet)")
    print(LINE)
    return 0


async def run_ingestion(args: argparse.Namespace) -> int:
    """Execute an incremental or backfill run."""
    async with IngestionPipeline() as pipeline:
        if args.backfill:
            print(
                f"Backfilling {args.category} from {args.since.date()}"
                f"{f' to {args.until.date()}' if args.until else ''}, "
                f"limit {args.limit}\n"
            )
            summary = await pipeline.run_backfill(
                category=args.category,
                since=args.since,
                until=args.until,
                limit=args.limit,
            )
        else:
            print(f"Incremental sync of {args.category}, limit {args.limit}\n")
            summary = await pipeline.run_incremental(category=args.category, limit=args.limit)

    print_summary(summary)
    # A run with some failures is partial, not a process failure -- some PDFs
    # are simply malformed. Only a run that achieved nothing signals a problem.
    return 0 if summary.fetched == 0 or summary.failed < summary.fetched else 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest arXiv papers into PostgreSQL and OpenSearch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--category", default="cs.CL", help="arXiv category (default: cs.CL)")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum papers to process (default: 5). Always bounded: arXiv asks "
        "for three seconds between requests.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Fetch a historical window instead of syncing from the checkpoint",
    )
    parser.add_argument("--since", type=_parse_date, help="Backfill start, YYYY-MM-DD")
    parser.add_argument("--until", type=_parse_date, help="Backfill end, YYYY-MM-DD")
    parser.add_argument("--status", action="store_true", help="Show pipeline state and exit")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    args = parser.parse_args()
    if args.backfill and args.since is None:
        parser.error("--backfill requires --since")
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


async def main_async(args: argparse.Namespace) -> int:
    """Entry point, ensuring pooled connections are closed."""
    settings = get_settings()
    configure_logging(level=args.log_level, log_format=settings.app.log_format)
    try:
        return await (show_status() if args.status else run_ingestion(args))
    finally:
        await dispose_engine()


def main() -> int:
    """Run the async entry point and translate failures into an exit code."""
    args = parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted. Progress so far is committed; re-run to continue.")
        return 130
    except Exception as exc:
        print(f"\nIngestion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
