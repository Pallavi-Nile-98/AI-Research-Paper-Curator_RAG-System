"""Async client for the arXiv API.

    from paper_curator.ingestion.arxiv_client import ArxivClient, ArxivQuery

    query = ArxivQuery(categories=["cs.CL", "cs.IR"])
    async with ArxivClient() as client:
        async for paper in client.iter_papers(query, limit=50):
            ...

This package fetches and validates. It does not decide what is new, what to keep
or where to resume — that is the pipeline layer's job, which keeps this testable
with no database and keeps the pipeline testable with no network.
"""

from paper_curator.ingestion.arxiv_client.client import MAX_PAGE_SIZE, ArxivClient
from paper_curator.ingestion.arxiv_client.errors import ArxivApiError, ArxivParseError
from paper_curator.ingestion.arxiv_client.models import ArxivPage, ArxivPaper
from paper_curator.ingestion.arxiv_client.parser import (
    parse_entry,
    parse_feed,
    split_versioned_id,
)
from paper_curator.ingestion.arxiv_client.query import ArxivQuery, SortBy, SortOrder

__all__ = [
    "MAX_PAGE_SIZE",
    "ArxivApiError",
    "ArxivClient",
    "ArxivPage",
    "ArxivPaper",
    "ArxivParseError",
    "ArxivQuery",
    "SortBy",
    "SortOrder",
    "parse_entry",
    "parse_feed",
    "split_versioned_id",
]
