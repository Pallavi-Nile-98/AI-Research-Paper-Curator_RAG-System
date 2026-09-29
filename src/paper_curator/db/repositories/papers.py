"""Persistence for papers, authors, extracted text and chunks.

The interesting function here is :func:`upsert_paper`, which decides what a
freshly fetched paper *means* relative to what is already stored. There are
three possibilities and they must be distinguished, because they lead to
different work:

* **Already have this exact version.** Skip it. Re-downloading a 2 MB PDF and
  re-embedding forty chunks to arrive at identical rows is pure waste, and at
  arXiv's three-second rate limit it is the difference between a five-minute
  incremental run and an hour-long one.
* **Have an older version.** Store the new one, mark it latest, and demote the
  old. The old version's rows stay -- a citation should still resolve to the
  version it was made against.
* **Never seen it.** Store it.

The database enforces the invariants underneath all of this: a unique constraint
on ``(arxiv_id, version)`` and a partial unique index allowing one ``is_latest``
row per identifier. This code cooperates with those constraints rather than
substituting for them.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from paper_curator.core.logging import get_logger
from paper_curator.db.enums import ExtractionMethod
from paper_curator.db.models import (
    Author,
    Chunk,
    DocumentText,
    Paper,
    PaperAuthor,
)
from paper_curator.ingestion.arxiv_client.models import ArxivPaper
from paper_curator.ingestion.chunking.models import TextChunk

logger = get_logger(__name__)

_PUNCTUATION = re.compile(r"[^\w\s]")


class UpsertOutcome(StrEnum):
    """What storing a fetched paper actually did."""

    CREATED = "created"
    """First time we have seen this identifier."""

    UPDATED = "updated"
    """A newer version replaced an older one, which was demoted."""

    SKIPPED = "skipped"
    """This exact version was already stored; nothing to do."""


@dataclass(frozen=True, slots=True)
class UpsertResult:
    """The stored paper and what happened to it."""

    paper: Paper
    outcome: UpsertOutcome

    @property
    def needs_processing(self) -> bool:
        """Whether the PDF pipeline should run for this paper.

        A skipped paper already has its text, chunks and index entries.
        """
        return self.outcome is not UpsertOutcome.SKIPPED


def normalize_author_name(name: str) -> str:
    """Reduce a name to a key for deduplication.

    Best-effort and explicitly not authoritative. Author disambiguation is a
    hard problem -- "J. Smith", "John Smith" and "John A. Smith" may or may not
    be one person, and no normalisation answers that. This collapses case,
    punctuation and spacing so obvious duplicates merge, and accepts that it
    will occasionally merge two people who share a name.
    """
    return " ".join(_PUNCTUATION.sub(" ", name).lower().split())


async def get_or_create_authors(session: AsyncSession, names: list[str]) -> list[Author]:
    """Fetch existing authors by normalised name, creating any that are new.

    Uses ``ON CONFLICT DO NOTHING`` rather than a check-then-insert. Two
    concurrent ingestion runs sharing an author would otherwise both find
    nothing, both insert, and one would fail on the unique constraint.
    """
    if not names:
        return []

    # Preserve order and drop duplicates within this paper's own author list.
    seen: dict[str, str] = {}
    for name in names:
        key = normalize_author_name(name)
        if key and key not in seen:
            seen[key] = name

    if not seen:
        return []

    await session.execute(
        pg_insert(Author)
        .values([{"full_name": full, "normalized_name": key} for key, full in seen.items()])
        .on_conflict_do_nothing(index_elements=["normalized_name"]),
    )

    result = await session.execute(select(Author).where(Author.normalized_name.in_(seen.keys())))
    by_key = {author.normalized_name: author for author in result.scalars()}
    return [by_key[key] for key in seen if key in by_key]


async def upsert_paper(session: AsyncSession, fetched: ArxivPaper) -> UpsertResult:
    """Store a fetched paper, resolving it against any version already held."""
    existing_versions = (
        (
            await session.execute(
                select(Paper).where(Paper.arxiv_id == fetched.arxiv_id).order_by(Paper.version)
            )
        )
        .scalars()
        .all()
    )

    for stored in existing_versions:
        if stored.version == fetched.version:
            logger.debug("paper_skipped", arxiv_id=fetched.arxiv_id, version=fetched.version)
            return UpsertResult(paper=stored, outcome=UpsertOutcome.SKIPPED)

    is_newest = all(stored.version < fetched.version for stored in existing_versions)

    if existing_versions and is_newest:
        # Demote every prior version in one statement, before inserting the new
        # one. Doing it afterwards would momentarily have two rows flagged
        # latest, which the partial unique index rejects.
        await session.execute(
            update(Paper)
            .where(Paper.arxiv_id == fetched.arxiv_id, Paper.is_latest.is_(True))
            .values(is_latest=False)
        )

    paper = Paper(
        arxiv_id=fetched.arxiv_id,
        version=fetched.version,
        is_latest=is_newest,
        title=fetched.title,
        abstract=fetched.abstract,
        published_at=fetched.published_at,
        updated_at_source=fetched.updated_at,
        primary_category=fetched.primary_category,
        categories=list(fetched.categories),
        doi=fetched.doi,
        journal_ref=fetched.journal_ref,
        comment=fetched.comment,
        abs_url=fetched.abs_url,
        pdf_url=fetched.pdf_url,
    )
    session.add(paper)
    await session.flush()

    authors = await get_or_create_authors(session, list(fetched.authors))
    for position, author in enumerate(authors):
        session.add(PaperAuthor(paper_id=paper.id, author_id=author.id, position=position))
    await session.flush()

    outcome = UpsertOutcome.UPDATED if existing_versions else UpsertOutcome.CREATED
    logger.info(
        "paper_stored",
        arxiv_id=fetched.arxiv_id,
        version=fetched.version,
        outcome=outcome.value,
        authors=len(authors),
    )
    return UpsertResult(paper=paper, outcome=outcome)


async def save_document_text(
    session: AsyncSession,
    *,
    paper_id: int,
    text: str,
    method: ExtractionMethod,
    page_count: int,
    quality_score: float,
) -> DocumentText:
    """Store or replace a paper's extracted text.

    Keeping the full text is what makes re-chunking cheap: tuning chunk size in
    Phase 2 reads this column instead of re-downloading every PDF at three
    seconds per request.
    """
    existing = await session.get(DocumentText, paper_id)
    if existing is not None:
        existing.full_text = text
        existing.extraction_method = method
        existing.page_count = page_count
        existing.char_count = len(text)
        existing.quality_score = quality_score
        return existing

    document = DocumentText(
        paper_id=paper_id,
        full_text=text,
        extraction_method=method,
        page_count=page_count,
        char_count=len(text),
        quality_score=quality_score,
    )
    session.add(document)
    await session.flush()
    return document


async def replace_chunks(
    session: AsyncSession,
    *,
    paper_id: int,
    arxiv_id: str,
    version: int,
    chunks: list[TextChunk],
    embedding_model: str,
) -> list[Chunk]:
    """Replace all of a paper's chunks with a freshly computed set.

    Delete-then-insert rather than a merge. Re-chunking with different settings
    changes how many chunks there are and where their boundaries fall, so
    matching old rows to new ones is not meaningful -- and leaving orphans
    behind would keep stale passages searchable.

    ``indexed_at`` is deliberately left NULL. A chunk counts as indexed only
    once OpenSearch acknowledges the write, and that gap is what makes
    dual-write drift detectable.
    """
    existing = (
        (await session.execute(select(Chunk).where(Chunk.paper_id == paper_id))).scalars().all()
    )
    for stale in existing:
        await session.delete(stale)
    await session.flush()

    rows = [
        Chunk(
            paper_id=paper_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            content_hash=chunk.content_hash,
            section=chunk.section,
            subsection=chunk.subsection,
            token_count=chunk.token_count,
            char_count=chunk.char_count,
            opensearch_doc_id=f"{arxiv_id}v{version}:{chunk.chunk_index}",
            embedding_model=embedding_model,
            indexed_at=None,
        )
        for chunk in chunks
    ]
    session.add_all(rows)
    await session.flush()

    logger.debug("chunks_stored", paper_id=paper_id, replaced=len(existing), stored=len(rows))
    return rows


async def mark_chunks_indexed(session: AsyncSession, document_ids: list[str]) -> int:
    """Record that OpenSearch acknowledged these chunks.

    Called with only the ids the bulk write actually accepted. Everything else
    keeps ``indexed_at IS NULL`` and is retried by a later run, which is the
    reconciliation mechanism between the two stores.
    """
    if not document_ids:
        return 0

    result = await session.execute(
        update(Chunk)
        .where(Chunk.opensearch_doc_id.in_(document_ids))
        .values(indexed_at=dt.datetime.now(dt.UTC))
    )
    # execute() is typed as returning Result, but a DML statement always yields
    # a CursorResult, which is what carries rowcount.
    return int(cast("CursorResult[Any]", result).rowcount or 0)


async def count_unindexed_chunks(session: AsyncSession) -> int:
    """Chunks present in PostgreSQL that OpenSearch has not confirmed.

    A non-zero count means the two stores disagree -- either a run is in
    progress, or a previous one failed partway.
    """
    result = await session.execute(select(Chunk.id).where(Chunk.indexed_at.is_(None)))
    return len(result.scalars().all())
