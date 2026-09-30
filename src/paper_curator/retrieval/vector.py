"""Dense vector retrieval via approximate nearest neighbours.

An embedding model maps the query and every chunk into a shared space where
proximity approximates semantic relatedness. A chunk can therefore be retrieved
without sharing a single word with the query, which is precisely where BM25
fails.

**Where this fails instead:** precise, rare or out-of-distribution tokens.
``arXiv:2401.12345`` has no meaningful embedding -- its vector sits near other
identifier-shaped strings, not near the paper it names. A technical term the
model saw rarely during training retrieves passages that are *generally about
the area* rather than the one that defines it. Embeddings compress, and
compression discards exactly the precision these queries depend on.

Two details matter more than they look.

**The query is embedded through the query path**, not the document path. The
default model is asymmetric: queries take an instruction prefix, passages do
not. Using the wrong one degrades retrieval and degrades it silently.

**Filters are applied inside the k-NN search**, which the Lucene engine
supports. Filtering afterwards would discard members of the top k and return
fewer results than asked for, with nothing to distinguish that from there
simply being fewer matches.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import OpenSearchException

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.logging import get_logger
from paper_curator.embedding import EmbeddingProvider
from paper_curator.retrieval.models import (
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
    RetrieverContribution,
    RetrieverKind,
)
from paper_curator.search.client import OpenSearchError

logger = get_logger(__name__)


class VectorRetriever:
    """Finds chunks by embedding similarity."""

    def __init__(
        self,
        client: AsyncOpenSearch,
        embedder: EmbeddingProvider,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._client = client
        self._embedder = embedder
        self._alias = resolved.opensearch.index_alias
        self._config = resolved.retrieval

    def build_query(
        self, vector: list[float], size: int, filters: RetrievalFilters | None = None
    ) -> dict[str, Any]:
        """Build the k-NN query body."""
        knn: dict[str, Any] = {"vector": vector, "k": size}

        if filters and not filters.is_empty:
            # Evaluated during graph traversal by the Lucene engine, so the
            # result is the top k *among matching documents* rather than the
            # top k overall with non-matches removed.
            knn["filter"] = {"bool": {"filter": filters.to_opensearch_clauses()}}

        return {"query": {"knn": {"embedding": knn}}}

    async def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        """Return chunks ranked by embedding similarity to the query."""
        size = limit or self._config.candidate_pool_size

        # Embedding is CPU-bound; keep it off the event loop. Timed separately
        # because on a machine without a GPU it is a meaningful share of total
        # latency, and folding it into the search time would misattribute it.
        embed_started = time.perf_counter()
        vector = await asyncio.to_thread(self._embedder.embed_query, query)
        embedding_ms = int((time.perf_counter() - embed_started) * 1000)

        body = self.build_query(vector, size, filters)
        search_started = time.perf_counter()

        try:
            response = await self._client.search(index=self._alias, body=body, size=size)
        except OpenSearchException as exc:
            msg = f"vector retrieval failed: {exc}"
            raise OpenSearchError(msg, context={"query": query}) from exc

        search_ms = int((time.perf_counter() - search_started) * 1000)
        hits = response.get("hits", {}).get("hits", [])

        chunks = []
        for rank, hit in enumerate(hits, start=1):
            chunk = RetrievedChunk.from_hit(hit)
            chunks.append(
                replace(
                    chunk,
                    contributions=[
                        RetrieverContribution(
                            retriever=RetrieverKind.VECTOR, rank=rank, score=chunk.score
                        )
                    ],
                )
            )

        logger.debug(
            "vector_retrieval",
            query=query,
            returned=len(chunks),
            requested=size,
            embed_ms=embedding_ms,
            search_ms=search_ms,
        )
        return RetrievalResult(
            chunks=chunks,
            mode=RetrieverKind.VECTOR,
            query=query,
            total_candidates=len(chunks),
            retrieval_ms=embedding_ms + search_ms,
            embedding_ms=embedding_ms,
            filters_applied=bool(filters and not filters.is_empty),
        )
