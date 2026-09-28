"""Async HTTP client for the arXiv API.

Responsibilities are deliberately narrow: issue requests, respect the rate
limit, retry transient failures, and yield validated papers. It knows nothing
about the database, deduplication or checkpoints — those belong to the pipeline
layer, which can then be tested without any HTTP at all.

The HTTP client is injected rather than constructed internally, so tests supply
a mock transport and never touch the network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Self

import httpx

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.logging import get_logger
from paper_curator.core.retry import AsyncRateLimiter, retry_async
from paper_curator.ingestion.arxiv_client.errors import ArxivApiError, ArxivParseError
from paper_curator.ingestion.arxiv_client.models import ArxivPage, ArxivPaper
from paper_curator.ingestion.arxiv_client.parser import parse_feed
from paper_curator.ingestion.arxiv_client.query import ArxivQuery

logger = get_logger(__name__)

# arXiv rejects larger pages outright.
MAX_PAGE_SIZE = 2000

# Identifies this client in arXiv's logs. A generic default user agent is what
# gets blocked first when a service decides to shed traffic.
USER_AGENT = (
    "ai-research-paper-curator/0.1 "
    "(+https://github.com/Pallavi-Nile-98/AI-Research-Paper-Curator_RAG-System)"
)


class ArxivClient:
    """Fetches papers from the arXiv API.

    Use as an async context manager so the underlying connection pool is closed:

        async with ArxivClient() as client:
            async for paper in client.iter_papers(query, limit=100):
                ...
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        cfg = self._settings.arxiv

        self._base_url = cfg.base_url
        self._page_size = min(cfg.page_size, MAX_PAGE_SIZE)
        self._max_retries = cfg.max_retries

        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.request_timeout_seconds),
            follow_redirects=True,
        )
        self._rate_limiter = rate_limiter or AsyncRateLimiter(cfg.min_request_interval_seconds)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client, but only if this instance created it.

        Closing an injected client would be a surprising side effect for a
        caller that intends to keep using it.
        """
        if self._owns_client:
            await self._http.aclose()

    async def _request(self, params: dict[str, str | int]) -> str:
        """Issue one rate-limited request and return the response body."""
        await self._rate_limiter.acquire()

        try:
            # Set per request rather than on the client, so an injected client
            # still identifies itself. A generic "python-httpx/x.y" agent is the
            # first thing a service blocks when it decides to shed traffic.
            response = await self._http.get(
                self._base_url, params=params, headers={"User-Agent": USER_AGENT}
            )
        except httpx.TimeoutException as exc:
            msg = f"arXiv request timed out: {exc}"
            raise ArxivApiError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"arXiv request failed: {exc}"
            raise ArxivApiError(msg) from exc

        if response.status_code != httpx.codes.OK:
            msg = f"arXiv returned HTTP {response.status_code}"
            raise ArxivApiError(msg, status_code=response.status_code)

        return response.text

    async def fetch_page(
        self,
        query: ArxivQuery,
        *,
        start: int = 0,
        page_size: int | None = None,
    ) -> ArxivPage:
        """Fetch one page of results, retrying transient failures.

        Raises:
            ArxivApiError: The request failed and retries were exhausted, or it
                failed in a way retrying cannot fix.
            ArxivParseError: The response body was not a parseable feed.

        """
        size = min(page_size or self._page_size, MAX_PAGE_SIZE)
        params = query.to_params(start=start, max_results=size)

        async def attempt() -> str:
            try:
                return await self._request(params)
            except ArxivApiError as exc:
                # Non-retryable failures are re-raised as a type the retry
                # policy does not catch, so a 400 fails immediately instead of
                # being reissued three times.
                if not exc.is_retryable:
                    raise _PermanentArxivError(exc) from exc
                raise

        try:
            body = await retry_async(
                attempt,
                max_attempts=self._max_retries + 1,
                retry_on=(ArxivApiError,),
                operation_name="arxiv.fetch_page",
            )
        except _PermanentArxivError as wrapper:
            raise wrapper.original from wrapper

        page = parse_feed(body)

        skipped = max(0, min(page.items_per_page, size) - len(page.papers))
        logger.info(
            "arxiv_page_fetched",
            start=start,
            page_size=size,
            returned=len(page.papers),
            skipped_unparseable=skipped,
            total_results=page.total_results,
        )
        return page

    async def iter_papers(
        self,
        query: ArxivQuery,
        *,
        limit: int | None = None,
    ) -> AsyncIterator[ArxivPaper]:
        """Yield papers matching ``query``, paging until exhausted or ``limit``.

        ``limit`` bounds the work. Without it a broad query can page through tens
        of thousands of results, and at three seconds per request that is hours
        of wall-clock time — which is why backfill is always bounded.
        """
        if limit is not None and limit < 1:
            msg = f"limit must be at least 1, got {limit}"
            raise ValueError(msg)

        yielded = 0
        start = 0

        while True:
            remaining = None if limit is None else limit - yielded
            size = self._page_size if remaining is None else min(self._page_size, remaining)

            page = await self.fetch_page(query, start=start, page_size=size)

            for paper in page.papers:
                yield paper
                yielded += 1
                if limit is not None and yielded >= limit:
                    logger.info("arxiv_iteration_complete", reason="limit_reached", total=yielded)
                    return

            # An empty page while the API still reports more results means a
            # transient hiccup rather than the end of the result set. Advancing
            # would silently skip records, so stop and let the caller resume
            # from its checkpoint.
            if page.is_empty:
                reason = "exhausted" if not page.has_more else "empty_page_before_end"
                if page.has_more:
                    logger.warning(
                        "arxiv_empty_page",
                        start=start,
                        total_results=page.total_results,
                        detail="stopping early; checkpoint will resume from here",
                    )
                logger.info("arxiv_iteration_complete", reason=reason, total=yielded)
                return

            if not page.has_more:
                logger.info("arxiv_iteration_complete", reason="exhausted", total=yielded)
                return

            start = page.next_start_index

    async def fetch_by_ids(self, arxiv_ids: Sequence[str]) -> list[ArxivPaper]:
        """Fetch specific papers by identifier.

        Used by manual runs and to re-check a single paper. Identifiers may
        include a version suffix; without one, arXiv returns the latest.
        """
        if not arxiv_ids:
            return []

        params: dict[str, str | int] = {
            "id_list": ",".join(arxiv_ids),
            "start": 0,
            "max_results": min(len(arxiv_ids), MAX_PAGE_SIZE),
        }

        async def attempt() -> str:
            try:
                return await self._request(params)
            except ArxivApiError as exc:
                if not exc.is_retryable:
                    raise _PermanentArxivError(exc) from exc
                raise

        try:
            body = await retry_async(
                attempt,
                max_attempts=self._max_retries + 1,
                retry_on=(ArxivApiError,),
                operation_name="arxiv.fetch_by_ids",
            )
        except _PermanentArxivError as wrapper:
            raise wrapper.original from wrapper

        page = parse_feed(body)
        logger.info("arxiv_fetch_by_ids", requested=len(arxiv_ids), returned=len(page.papers))
        return page.papers


class _PermanentArxivError(ArxivParseError):
    """Internal wrapper marking a failure the retry policy must not catch.

    ``retry_async`` decides what to retry by exception type, but whether an
    :class:`ArxivApiError` is retryable depends on its status code. Wrapping
    non-retryable ones in a type outside ``retry_on`` lets a 400 fail on the
    first attempt while a 503 is retried, without teaching the generic retry
    helper anything about HTTP.
    """

    def __init__(self, original: ArxivApiError) -> None:
        super().__init__(str(original), context=original.context)
        self.original = original
