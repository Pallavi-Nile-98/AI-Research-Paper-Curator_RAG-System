"""Hybrid retrieval: BM25 and dense vectors, fused.

The argument for combining them, restated from ADR-0004 because it is the one
worth being able to defend: **their failure modes are uncorrelated.** BM25 fails
when the wording differs but the meaning matches. Vector search fails when the
wording matches exactly but the token is rare or meaningless to the embedder.
Those are close to opposite conditions, which is what makes the ensemble worth
more than either part.

That remains a hypothesis until measured. This class exists alongside the two
single-mode retrievers precisely so all three are real, production
configurations and the comparison in Phase 2 is between code paths that
actually ship -- not between a real system and a stub of a baseline.

Both retrievers run concurrently. They hit the same cluster but are independent
requests, so the wall-clock cost of hybrid retrieval is roughly the slower of
the two rather than their sum.
"""

from __future__ import annotations

import asyncio
import time

from opensearchpy import AsyncOpenSearch

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.logging import get_logger
from paper_curator.embedding import EmbeddingProvider
from paper_curator.retrieval.fusion import (
    deduplicate_chunks,
    limit_per_paper,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from paper_curator.retrieval.keyword import KeywordRetriever
from paper_curator.retrieval.models import (
    RetrievalFilters,
    RetrievalResult,
    RetrieverKind,
)
from paper_curator.retrieval.vector import VectorRetriever

logger = get_logger(__name__)


class HybridRetriever:
    """Runs lexical and semantic retrieval together and fuses the results."""

    def __init__(
        self,
        client: AsyncOpenSearch,
        embedder: EmbeddingProvider,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._config = resolved.retrieval
        self._keyword = KeywordRetriever(client, resolved)
        self._vector = VectorRetriever(client, embedder, resolved)

    async def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        """Retrieve by both paths, fuse, then shape the final list."""
        config = self._config
        final_size = limit or config.top_k
        started = time.perf_counter()

        # Concurrent: two independent requests to the same cluster, so the cost
        # is the slower of the two rather than their sum.
        keyword_result, vector_result = await asyncio.gather(
            self._keyword.retrieve(query, limit=config.candidate_pool_size, filters=filters),
            self._vector.retrieve(query, limit=config.candidate_pool_size, filters=filters),
        )

        lists = [keyword_result.chunks, vector_result.chunks]

        if config.fusion_method == "rrf":
            fused = reciprocal_rank_fusion(lists, k=config.rrf_k)
        else:
            fused = weighted_score_fusion(
                lists, weights=[config.keyword_weight, config.vector_weight]
            )

        total_candidates = len(fused)

        # Shaping, in this order deliberately: remove near-duplicates first, so
        # the per-paper cap is spent on distinct passages rather than on two
        # copies of the same one.
        if config.deduplicate:
            fused = deduplicate_chunks(fused, threshold=config.duplicate_similarity_threshold)
        fused = limit_per_paper(fused, max_per_paper=config.max_chunks_per_paper)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        chunks = fused[:final_size]

        both = sum(1 for chunk in chunks if len(chunk.found_by) > 1)
        logger.info(
            "hybrid_retrieval",
            query=query,
            keyword_hits=len(keyword_result.chunks),
            vector_hits=len(vector_result.chunks),
            fused_candidates=total_candidates,
            returned=len(chunks),
            found_by_both=both,
            fusion=config.fusion_method,
            took_ms=elapsed_ms,
        )

        return RetrievalResult(
            chunks=chunks,
            mode=RetrieverKind.HYBRID,
            query=query,
            total_candidates=total_candidates,
            retrieval_ms=elapsed_ms,
            embedding_ms=vector_result.embedding_ms,
            filters_applied=bool(filters and not filters.is_empty),
        )
