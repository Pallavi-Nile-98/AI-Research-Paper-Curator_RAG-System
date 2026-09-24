"""Models for papers, authors, extracted text and chunks.

These are the *content* tables. Pipeline bookkeeping lives in ``pipeline.py`` and
user-facing records in ``feedback.py``.

The design follows ADR-0002: PostgreSQL is the system of record and OpenSearch is
a derived index. Concretely that means the extracted text and every chunk's text
are stored here, so the search index can be rebuilt — with different chunk sizes
or a different embedding model — without re-downloading a single PDF.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

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
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from paper_curator.db.base import Base, TimestampMixin
from paper_curator.db.enums import ExtractionMethod

if TYPE_CHECKING:
    from paper_curator.db.models.pipeline import DocumentProcessingStatus


class Paper(Base, TimestampMixin):
    """One *version* of one arXiv paper.

    arXiv papers are revised, and a revision can change the abstract, the text,
    and the conclusions. Each version is therefore its own row rather than an
    update in place, so a citation can point at the version that was actually
    retrieved.
    """

    __tablename__ = "papers"
    __table_args__ = (
        # The idempotency guarantee for the whole ingestion pipeline. Re-running
        # ingestion over the same window cannot create duplicates, because the
        # database refuses them -- rather than relying on application logic
        # being correct on every code path.
        UniqueConstraint("arxiv_id", "version", name="uq_papers_arxiv_id_version"),
        # At most one row per arXiv ID may be flagged latest. A partial unique
        # index enforces this in the database; maintaining `is_latest` purely in
        # application code would eventually drift.
        Index(
            "uq_papers_arxiv_id_latest",
            "arxiv_id",
            unique=True,
            postgresql_where=text("is_latest"),
        ),
        Index("ix_papers_published_at", "published_at"),
        Index("ix_papers_primary_category", "primary_category"),
        CheckConstraint("version >= 1", name="version_positive"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)

    arxiv_id: Mapped[str] = mapped_column(
        String(32), nullable=False, doc="arXiv identifier without version, e.g. 2401.12345"
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, doc="arXiv version, 1-based")
    is_latest: Mapped[bool] = mapped_column(
        nullable=False, default=True, doc="True for the highest known version of this arxiv_id"
    )

    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)

    # arXiv exposes two dates: when v1 appeared, and when this version appeared.
    # Incremental sync watermarks on updated_at_source, because a revision to an
    # old paper must be picked up even though its published_at is long past.
    published_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at_source: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    primary_category: Mapped[str] = mapped_column(String(64), nullable=False)
    categories: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, doc="All arXiv categories, primary first"
    )

    doi: Mapped[str | None] = mapped_column(String(255))
    journal_ref: Mapped[str | None] = mapped_column(Text)
    comment: Mapped[str | None] = mapped_column(Text)

    abs_url: Mapped[str] = mapped_column(Text, nullable=False, doc="Human-readable arXiv page")
    pdf_url: Mapped[str] = mapped_column(Text, nullable=False)

    authors: Mapped[list[PaperAuthor]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
        order_by="PaperAuthor.position",
    )
    document_text: Mapped[DocumentText | None] = relationship(
        back_populates="paper", cascade="all, delete-orphan", uselist=False
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", order_by="Chunk.chunk_index"
    )
    processing_status: Mapped[DocumentProcessingStatus | None] = relationship(
        back_populates="paper", cascade="all, delete-orphan", uselist=False
    )

    def __repr__(self) -> str:
        return f"<Paper {self.arxiv_id}v{self.version}>"


class Author(Base, TimestampMixin):
    """A distinct author, deduplicated across papers.

    Author disambiguation is genuinely hard — the same person appears as
    "J. Smith", "John Smith" and "John A. Smith". ``normalized_name`` is a
    best-effort key (lowercased, punctuation stripped) and is deliberately
    *not* presented as authoritative. Treating it as exact identity would
    silently merge distinct people who share a name.
    """

    __tablename__ = "authors"
    __table_args__ = (UniqueConstraint("normalized_name", name="uq_authors_normalized_name"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    full_name: Mapped[str] = mapped_column(Text, nullable=False, doc="Name as arXiv supplied it")
    normalized_name: Mapped[str] = mapped_column(
        Text, nullable=False, doc="Lowercased, punctuation-stripped key used only for dedup"
    )

    papers: Mapped[list[PaperAuthor]] = relationship(back_populates="author")

    def __repr__(self) -> str:
        return f"<Author {self.full_name!r}>"


class PaperAuthor(Base):
    """Association between a paper version and an author, preserving order.

    Author order carries meaning in academic publishing — first author and last
    author are not interchangeable — so this is an explicit association object
    with a ``position`` column rather than a plain many-to-many table.
    """

    __tablename__ = "paper_authors"
    __table_args__ = (
        UniqueConstraint("paper_id", "position", name="uq_paper_authors_paper_id_position"),
        CheckConstraint("position >= 0", name="position_non_negative"),
    )

    paper_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    author_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("authors.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, doc="0-based author order")

    paper: Mapped[Paper] = relationship(back_populates="authors")
    author: Mapped[Author] = relationship(back_populates="papers")


class DocumentText(Base, TimestampMixin):
    """Full extracted text of a paper's PDF, stored once.

    Kept in its own table rather than as a column on ``papers`` so that listing
    or filtering papers never drags tens of kilobytes of body text through the
    query.

    Storing it at all is what makes re-chunking cheap. Tuning chunk size in
    Phase 2 means re-reading this column, not re-downloading and re-parsing
    every PDF — which at three seconds per arXiv request would make the
    experiment impractical.
    """

    __tablename__ = "document_texts"

    paper_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_method: Mapped[ExtractionMethod] = mapped_column(
        Enum(ExtractionMethod, native_enum=False, length=16), nullable=False
    )
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_score: Mapped[float] = mapped_column(
        nullable=False,
        doc="0-1 heuristic confidence in the extracted text; drives the OCR fallback",
    )

    paper: Mapped[Paper] = relationship(back_populates="document_text")


class Chunk(Base, TimestampMixin):
    """One retrievable passage of a paper.

    Chunk text lives here as well as in OpenSearch. The duplication is
    deliberate: ADR-0002 makes PostgreSQL the system of record, so losing the
    search cluster must be recoverable by re-indexing rather than by
    re-ingesting.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("paper_id", "chunk_index", name="uq_chunks_paper_id_chunk_index"),
        # Deterministic OpenSearch document ID. Re-indexing the same chunk
        # overwrites rather than duplicating, which is what makes the indexing
        # step safe to retry after a partial failure.
        UniqueConstraint("opensearch_doc_id", name="uq_chunks_opensearch_doc_id"),
        # Finds chunks whose text changed after re-extraction, and chunks that
        # were never confirmed indexed.
        Index("ix_chunks_content_hash", "content_hash"),
        Index("ix_chunks_indexed_at", "indexed_at"),
        CheckConstraint("chunk_index >= 0", name="chunk_index_non_negative"),
        CheckConstraint("token_count > 0", name="token_count_positive"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    paper_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, doc="0-based, ordered")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="SHA-256 of the chunk text; detects drift between Postgres and OpenSearch",
    )

    section: Mapped[str | None] = mapped_column(Text, doc="e.g. 'Methods'")
    subsection: Mapped[str | None] = mapped_column(Text, doc="e.g. 'Training details'")

    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)

    opensearch_doc_id: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(
        String(128), doc="Which model produced the stored vector; changing it forces a reindex"
    )
    # NULL until OpenSearch acknowledges the bulk write. This is the marker that
    # makes dual-write reconciliation possible: a chunk that exists here with a
    # NULL indexed_at is one Postgres committed but the index may not have.
    indexed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    paper: Mapped[Paper] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk paper={self.paper_id} #{self.chunk_index}>"
