"""BM25 lexical retrieval.

BM25 scores a document by the query terms it literally contains, weighted by
how rare each term is across the corpus, normalised for document length and
with term-frequency saturation so repetition has diminishing returns. It is
Lucene's default ranking function, and therefore OpenSearch's.

**What it is good at:** exact terminology. A query naming BM25, arXiv:2401.12345
or a specific author matches documents containing those tokens, regardless of
whether any model has ever seen the term.

**Where it fails:** vocabulary mismatch. A user asking how a search system
decides which words matter shares almost no tokens with a paper that says
inverse document frequency weighting. The paper is exactly on topic and BM25
scores it near zero. That failure is the reason the vector retriever exists.

The query searches three fields at different weights, because the index stores
text twice -- stemmed for recall, unstemmed for precision -- and a match in the
title means something different from a match in the body.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import OpenSearchException

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.logging import get_logger
from paper_curator.retrieval.models import (
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
    RetrieverContribution,
    RetrieverKind,
)
from paper_curator.search.client import OpenSearchError

logger = get_logger(__name__)


class KeywordRetriever:
    """Finds chunks by lexical overlap, scored with BM25."""

    def __init__(self, client: AsyncOpenSearch, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        self._client = client
        self._alias = resolved.opensearch.index_alias
        self._config = resolved.retrieval

    def build_query(self, query: str, filters: RetrievalFilters | None = None) -> dict[str, Any]:
        """Build the OpenSearch query body.

        Exposed separately from :meth:`retrieve` so the query can be inspected
        and tested without a cluster -- and so an unexpected result can be
        reproduced by hand in Dashboards.
        """
        config = self._config
        should: list[dict[str, Any]] = [
            # Stemmed: recall. "retrieving" matches "retrieval".
            {"match": {"text": {"query": query, "boost": config.bm25_text_boost}}},
            # Unstemmed: precision. Keeps technical terms and identifiers intact.
            {"match": {"text.exact": {"query": query, "boost": config.bm25_exact_boost}}},
            # A title match suggests the whole paper is about the query, not
            # that the phrase merely occurs somewhere in it.
            {"match": {"title": {"query": query, "boost": config.bm25_title_boost}}},
        ]

        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "should": should,
                    # At least one clause must match, otherwise a pure-filter
                    # query would return everything with a score of zero.
                    "minimum_should_match": 1,
                }
            }
        }

        if filters and not filters.is_empty:
            # Filter context: constrains eligibility without contributing to
            # the relevance score.
            body["query"]["bool"]["filter"] = filters.to_opensearch_clauses()

        return body

    async def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        """Return chunks ranked by BM25 score."""
        size = limit or self._config.candidate_pool_size
        body = self.build_query(query, filters)
        started = time.perf_counter()

        try:
            response = await self._client.search(index=self._alias, body=body, size=size)
        except OpenSearchException as exc:
            msg = f"keyword retrieval failed: {exc}"
            raise OpenSearchError(msg, context={"query": query}) from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        hits = response.get("hits", {}).get("hits", [])

        chunks = []
        for rank, hit in enumerate(hits, start=1):
            chunk = RetrievedChunk.from_hit(hit)
            chunks.append(
                replace(
                    chunk,
                    contributions=[
                        RetrieverContribution(
                            retriever=RetrieverKind.KEYWORD, rank=rank, score=chunk.score
                        )
                    ],
                )
            )

        logger.debug(
            "keyword_retrieval",
            query=query,
            returned=len(chunks),
            requested=size,
            took_ms=elapsed_ms,
        )
        return RetrievalResult(
            chunks=chunks,
            mode=RetrieverKind.KEYWORD,
            query=query,
            total_candidates=len(chunks),
            retrieval_ms=elapsed_ms,
            filters_applied=bool(filters and not filters.is_empty),
        )
