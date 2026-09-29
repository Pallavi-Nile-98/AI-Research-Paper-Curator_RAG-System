"""Chunk data structures.

A chunk is split into two layers on purpose.

:class:`TextChunk` is what the chunker produces: the passage and where it came
from *within the paper*. It knows nothing about arXiv, so the chunking algorithm
can be tested against plain strings with no paper object in sight.

:func:`build_chunk_document` then attaches the paper's identity -- arXiv ID,
version, title, authors, dates, URLs -- producing the document that goes into
OpenSearch. That metadata has to travel with every chunk because retrieval
returns chunks, not papers: a passage with no idea which paper it came from
cannot be cited, and citation is the point of the whole system.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from typing import Any

from paper_curator.ingestion.chunking.sections import SectionKind


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One retrievable passage, with its position in the paper's structure."""

    text: str
    chunk_index: int
    token_count: int

    section: str | None = None
    """Heading of the enclosing section, e.g. "Method"."""

    subsection: str | None = None
    """Heading of the subsection, when the chunk sits inside one."""

    kind: SectionKind = SectionKind.BODY

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def content_hash(self) -> str:
        """SHA-256 of the exact chunk text.

        Two jobs. It makes the OpenSearch document ID deterministic, so
        re-indexing overwrites rather than duplicates. And it detects drift: if
        a paper is re-extracted and a chunk's hash changes, the indexed copy is
        stale and must be replaced. Both depend on hashing the text exactly as
        stored -- normalising first would make a real change invisible.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def location(self) -> str:
        """Human-readable position, for citations and debugging."""
        if self.subsection and self.section:
            return f"{self.section} > {self.subsection}"
        return self.section or self.kind.value

    def __repr__(self) -> str:
        preview = self.text[:40].replace("\n", " ")
        return f"<TextChunk #{self.chunk_index} {self.location!r} {self.token_count}t {preview!r}>"


def build_chunk_document(
    chunk: TextChunk,
    *,
    arxiv_id: str,
    version: int,
    title: str,
    authors: list[str],
    abstract: str,
    published_at: dt.datetime,
    updated_at: dt.datetime,
    primary_category: str,
    categories: list[str],
    abs_url: str,
    pdf_url: str,
) -> dict[str, Any]:
    """Combine a chunk with its paper's metadata into an indexable document.

    The returned shape is what gets written to OpenSearch and what retrieval
    reads back, so every field needed to render a citation or apply a filter has
    to be present here. Denormalising paper metadata onto every chunk is
    deliberate: a search index is denormalised so filters can be applied during
    scoring rather than after it (ADR-0002).
    """
    return {
        # Deterministic identity: re-indexing the same chunk overwrites it.
        "chunk_id": f"{arxiv_id}v{version}:{chunk.chunk_index}",
        "content_hash": chunk.content_hash,
        # --- The retrievable content ---
        "text": chunk.text,
        # --- Position within the paper ---
        "chunk_index": chunk.chunk_index,
        "section": chunk.section,
        "subsection": chunk.subsection,
        "section_kind": chunk.kind.value,
        "token_count": chunk.token_count,
        "char_count": chunk.char_count,
        # --- Paper identity, for citation ---
        "arxiv_id": arxiv_id,
        "version": version,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        # --- Filterable metadata ---
        "published_at": published_at.isoformat(),
        "updated_at": updated_at.isoformat(),
        "primary_category": primary_category,
        "categories": categories,
        # --- Links shown to the reader ---
        "abs_url": abs_url,
        "pdf_url": pdf_url,
    }
