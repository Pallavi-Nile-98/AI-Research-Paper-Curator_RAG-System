"""Fetch real papers from arXiv and print them.

A manual smoke test for the ingestion client. The automated test suite mocks
every network call, so this is the one place the real API is exercised.

    python scripts/demo_arxiv.py
    python scripts/demo_arxiv.py --categories cs.CL cs.IR --limit 5
    python scripts/demo_arxiv.py --keywords "retrieval augmented generation"
    python scripts/demo_arxiv.py --ids 1706.03762 2005.11401

This talks to the live arXiv API. The client waits three seconds between
requests, as arXiv's terms of use ask, so a run takes a few seconds even for a
handful of papers. That delay is deliberate -- please do not remove it.

Nothing is written to a database; this only reads and prints.
"""

from __future__ import annotations

import argparse
import asyncio
import textwrap

from paper_curator.core.config import get_settings
from paper_curator.core.logging import configure_logging
from paper_curator.ingestion.arxiv_client import (
    ArxivClient,
    ArxivPaper,
    ArxivQuery,
    SortOrder,
)

LINE = "-" * 78


def render(paper: ArxivPaper, index: int) -> str:
    """Format one paper for the terminal."""
    authors = ", ".join(paper.authors[:4])
    if len(paper.authors) > 4:
        authors += f", +{len(paper.authors) - 4} more"

    abstract = textwrap.fill(paper.abstract, width=76, initial_indent="  ", subsequent_indent="  ")
    if len(abstract) > 600:
        abstract = abstract[:600].rstrip() + " ..."

    revised = ""
    if paper.updated_at.date() != paper.published_at.date():
        revised = f"  (revised {paper.updated_at.date()})"

    return "\n".join(
        [
            f"{index}. {paper.versioned_id}   [{paper.primary_category}]",
            f"   {textwrap.fill(paper.title, width=74, subsequent_indent='   ')}",
            f"   {authors}",
            f"   published {paper.published_at.date()}{revised}",
            f"   categories: {', '.join(paper.categories)}",
            f"   {paper.abs_url}",
            "",
            abstract,
            "",
        ]
    )


async def run(args: argparse.Namespace) -> int:
    """Fetch and print, returning a process exit code."""
    settings = get_settings()
    configure_logging(level=args.log_level, log_format=settings.app.log_format)

    async with ArxivClient(settings) as client:
        if args.ids:
            print(f"Fetching {len(args.ids)} paper(s) by identifier ...\n")
            papers = await client.fetch_by_ids(args.ids)
        else:
            # Newest first by default. The library default is ASCENDING,
            # because incremental sync must process oldest-first for a partial
            # run to leave a usable checkpoint -- but that makes a demo show
            # papers from 1994, which is confusing rather than instructive.
            query = ArxivQuery(
                categories=args.categories or [],
                keywords=args.keywords or [],
                sort_order=SortOrder.ASCENDING if args.oldest_first else SortOrder.DESCENDING,
            )
            print(f"Query : {query.to_search_query()}")
            print(f"Limit : {args.limit}")
            print("\nContacting arXiv (3s between requests, as their terms ask) ...\n")
            papers = [paper async for paper in client.iter_papers(query, limit=args.limit)]

    if not papers:
        print("No papers returned. Try a broader query.")
        return 1

    print(LINE)
    for index, paper in enumerate(papers, start=1):
        print(render(paper, index))
        print(LINE)

    print(f"\n{len(papers)} paper(s) fetched, parsed and validated.")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch real papers from the arXiv API and print them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cs.CL"],
        help="arXiv categories, e.g. cs.CL cs.IR (default: cs.CL)",
    )
    parser.add_argument("--keywords", nargs="+", help="Free-text terms to require")
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Fetch specific papers by identifier instead of searching",
    )
    parser.add_argument("--limit", type=int, default=3, help="Maximum papers (default: 3)")
    parser.add_argument(
        "--oldest-first",
        action="store_true",
        help="Sort as the ingestion pipeline does, oldest first (default: newest first)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Use DEBUG to see request and retry details",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    try:
        return asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
