"""Retrieval: finding the passages an answer will be built from.

    from paper_curator.retrieval import build_retriever

    retriever = build_retriever(client, embedder)
    result = await retriever.retrieve("what is BM25?")

Three interchangeable retrievers, selected by configuration:

* ``keyword`` -- BM25 lexical scoring. Strong on exact terminology, weak when
  the query's wording differs from the paper's.
* ``vector`` -- dense embedding similarity. The opposite profile: strong on
  paraphrase, weak on identifiers and rare technical terms.
* ``hybrid`` -- both, fused by rank.

All three are real configurations rather than a system plus two stubs, so the
Phase 2 comparison measures code paths that actually ship. See ADR-0004 for the
reasoning and for the commitment that no claim of hybrid superiority appears
anywhere until the recorded numbers support it.
"""

from opensearchpy import AsyncOpenSearch

from paper_curator.core.config import Settings, get_settings
from paper_curator.embedding import EmbeddingProvider
from paper_curator.retrieval.context import AssembledContext, Citation, ContextBuilder
from paper_curator.retrieval.fusion import (
    deduplicate_chunks,
    limit_per_paper,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from paper_curator.retrieval.hybrid import HybridRetriever
from paper_curator.retrieval.keyword import KeywordRetriever
from paper_curator.retrieval.models import (
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
    RetrieverContribution,
    RetrieverKind,
)
from paper_curator.retrieval.reranking import (
    CrossEncoderReranker,
    IdentityReranker,
    Reranker,
    RerankingService,
    RerankOutcome,
)
from paper_curator.retrieval.vector import VectorRetriever

__all__ = [
    "AssembledContext",
    "Citation",
    "ContextBuilder",
    "CrossEncoderReranker",
    "HybridRetriever",
    "IdentityReranker",
    "KeywordRetriever",
    "RerankOutcome",
    "Reranker",
    "RerankingService",
    "RetrievalFilters",
    "RetrievalResult",
    "RetrievedChunk",
    "RetrieverContribution",
    "RetrieverKind",
    "VectorRetriever",
    "build_retriever",
    "deduplicate_chunks",
    "limit_per_paper",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
]

Retriever = KeywordRetriever | VectorRetriever | HybridRetriever


def build_retriever(
    client: AsyncOpenSearch,
    embedder: EmbeddingProvider,
    settings: Settings | None = None,
    *,
    mode: str | None = None,
) -> Retriever:
    """Build the configured retriever.

    ``mode`` overrides the configured default, which is what lets an experiment
    sweep all three without rebuilding anything else.
    """
    resolved = settings or get_settings()
    selected = mode or resolved.retrieval.mode

    if selected == "keyword":
        return KeywordRetriever(client, resolved)
    if selected == "vector":
        return VectorRetriever(client, embedder, resolved)
    if selected == "hybrid":
        return HybridRetriever(client, embedder, resolved)

    msg = f"unknown retrieval mode: {selected!r}"
    raise ValueError(msg)
