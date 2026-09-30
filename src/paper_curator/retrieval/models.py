"""Types shared across the retrieval layer.

The design point here is **provenance**. A retrieved chunk carries not just a
score but where that score came from: which retriever returned it, at what
rank, with what raw score. Fusion then records how the combination was reached.

Without that, a bad result is undiagnosable. "Why did this irrelevant passage
rank third?" has a real answer -- it was ranked first by BM25 and absent from
the vector results, or vice versa -- and that answer is what makes tuning
possible rather than guesswork. It also feeds the debug panel in the UI and the
failure analysis in Phase 4.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RetrieverKind(StrEnum):
    """Which retrieval path produced a result."""

    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    """Metadata constraints applied during scoring.

    Applied inside the search engine, never to the results afterwards.
    Post-filtering a top-k list silently returns fewer than k documents: ask
    for 10, filter 6 away, get 4 -- with nothing to indicate that the other 6
    were discarded rather than never existing.
    """

    arxiv_ids: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    published_after: dt.datetime | None = None
    published_before: dt.datetime | None = None

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.arxiv_ids,
                self.authors,
                self.categories,
                self.published_after,
                self.published_before,
            )
        )

    def to_opensearch_clauses(self) -> list[dict[str, Any]]:
        """Render as OpenSearch filter clauses.

        Filter context rather than query context: these constrain *which*
        documents are eligible and must not influence relevance scoring. A
        category filter that contributed to the score would rank a paper higher
        merely for being in the requested category.
        """
        clauses: list[dict[str, Any]] = []

        if self.arxiv_ids:
            clauses.append({"terms": {"arxiv_id": self.arxiv_ids}})
        if self.categories:
            clauses.append({"terms": {"categories": self.categories}})
        if self.authors:
            # Matches the keyword subfield, so "Jane Doe" means that exact
            # author rather than any document mentioning Jane or Doe.
            clauses.append({"terms": {"authors.keyword": self.authors}})

        if self.published_after or self.published_before:
            bounds: dict[str, str] = {}
            if self.published_after:
                bounds["gte"] = self.published_after.isoformat()
            if self.published_before:
                bounds["lte"] = self.published_before.isoformat()
            clauses.append({"range": {"published_at": bounds}})

        return clauses


@dataclass(frozen=True, slots=True)
class RetrieverContribution:
    """One retriever's view of a chunk: where it placed it and why."""

    retriever: RetrieverKind
    rank: int
    """1-based position in that retriever's own result list."""

    score: float
    """Raw score, on that retriever's own scale.

    BM25 scores are unbounded and corpus-dependent; cosine similarity is
    bounded in [-1, 1]. They are kept unnormalised here precisely because they
    are not comparable -- which is the reason rank-based fusion exists.
    """


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with everything needed to cite it."""

    chunk_id: str
    text: str
    score: float
    """Final score, on whatever scale the producing retriever or fusion uses."""

    # --- Paper identity, for citation --------------------------------------
    arxiv_id: str
    version: int
    title: str
    authors: list[str]
    abs_url: str
    pdf_url: str
    published_at: str

    # --- Position within the paper ------------------------------------------
    chunk_index: int
    section: str | None = None
    subsection: str | None = None

    # --- Provenance -----------------------------------------------------------
    contributions: list[RetrieverContribution] = field(default_factory=list)
    """Which retrievers found this, where, and with what raw score."""

    rerank_score: float | None = None
    """Cross-encoder score, once re-ranking has run."""

    @property
    def versioned_id(self) -> str:
        return f"{self.arxiv_id}v{self.version}"

    @property
    def location(self) -> str:
        """Human-readable position, for a citation."""
        if self.section and self.subsection:
            return f"{self.section} > {self.subsection}"
        return self.section or "body"

    @property
    def found_by(self) -> list[RetrieverKind]:
        """Which retrievers surfaced this chunk.

        A chunk found by both is corroborated by two independent signals; one
        found by a single retriever may still be the right answer, and the
        distinction is exactly what fusion is weighing.
        """
        return [contribution.retriever for contribution in self.contributions]

    def explain(self) -> str:
        """One-line account of how this chunk was retrieved."""
        if not self.contributions:
            return f"score={self.score:.4f}"
        parts = [f"{c.retriever.value}#{c.rank}({c.score:.3f})" for c in self.contributions]
        line = f"score={self.score:.4f} via {' + '.join(parts)}"
        if self.rerank_score is not None:
            line += f" rerank={self.rerank_score:.4f}"
        return line

    @classmethod
    def from_hit(cls, hit: dict[str, Any]) -> RetrievedChunk:
        """Build from an OpenSearch search hit."""
        source = hit["_source"]
        return cls(
            chunk_id=str(hit["_id"]),
            text=source["text"],
            score=float(hit.get("_score") or 0.0),
            arxiv_id=source["arxiv_id"],
            version=int(source["version"]),
            title=source["title"],
            authors=list(source.get("authors") or []),
            abs_url=source.get("abs_url", ""),
            pdf_url=source.get("pdf_url", ""),
            published_at=source.get("published_at", ""),
            chunk_index=int(source.get("chunk_index", 0)),
            section=source.get("section"),
            subsection=source.get("subsection"),
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Retrieved chunks plus the diagnostics needed to explain the run."""

    chunks: list[RetrievedChunk]
    mode: RetrieverKind
    query: str

    total_candidates: int = 0
    """Candidates considered before fusion, deduplication and truncation."""

    retrieval_ms: int = 0
    rerank_ms: int | None = None
    embedding_ms: int | None = None
    """Time spent embedding the query.

    Reported separately because on a CPU-only machine it is a meaningful share
    of vector-retrieval latency, and attributing it to the search engine would
    make OpenSearch look slower than it is.
    """

    reranked: bool = False
    filters_applied: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def total_ms(self) -> int:
        return self.retrieval_ms + (self.rerank_ms or 0)

    def summary(self) -> str:
        """One-line description for logs."""
        timing = f"{self.retrieval_ms}ms"
        if self.embedding_ms is not None:
            timing += f" (embed {self.embedding_ms}ms)"
        if self.rerank_ms is not None:
            timing += f" + rerank {self.rerank_ms}ms"
        return (
            f"{self.mode.value}: {len(self.chunks)} of {self.total_candidates} "
            f"candidates in {timing}"
        )
