"""Tests for the async arXiv HTTP client.

Every test mocks the transport with respx, so nothing here touches the network.
That matters for more than speed: arXiv is a free service run by a university,
and a test suite that hits it on every CI run is exactly the behaviour its rate
limit exists to discourage.

The rate limiter is injected with a zero interval so tests do not wait three
seconds per request. The limiter itself is tested separately in test_retry.py.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from paper_curator.core.config import ArxivSettings, Settings
from paper_curator.core.retry import AsyncRateLimiter
from paper_curator.ingestion.arxiv_client import ArxivApiError, ArxivClient, ArxivQuery

FIXTURES = Path(__file__).parent.parent / "fixtures" / "arxiv"
API_URL = "http://export.arxiv.org/api/query"


def _feed(entries: str, *, total: int, start: int, per_page: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>{total}</opensearch:totalResults>
  <opensearch:startIndex>{start}</opensearch:startIndex>
  <opensearch:itemsPerPage>{per_page}</opensearch:itemsPerPage>
  {entries}
</feed>"""


def _entry(arxiv_id: str, version: int = 1) -> str:
    return f"""
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}v{version}</id>
    <updated>2024-01-01T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>Paper {arxiv_id}</title>
    <summary>Abstract for {arxiv_id}.</summary>
    <author><name>Test Author</name></author>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
  </entry>"""


def _settings(*, max_retries: int = 0, page_size: int = 100) -> Settings:
    """Build settings tuned for tests: no waiting, explicit retry budget."""
    return Settings(
        arxiv=ArxivSettings(
            max_retries=max_retries,
            page_size=page_size,
            min_request_interval_seconds=0,
        )
    )


@pytest.fixture
def query() -> ArxivQuery:
    return ArxivQuery(categories=["cs.CL"])


@pytest.fixture
async def client() -> ArxivClient:
    """Build a client with retries disabled and no rate limiting."""
    return ArxivClient(
        _settings(),
        http_client=httpx.AsyncClient(),
        rate_limiter=AsyncRateLimiter(0),
    )


@pytest.mark.unit
class TestFetchPage:
    @respx.mock
    async def test_returns_parsed_papers(self, client: ArxivClient, query: ArxivQuery) -> None:
        body = (FIXTURES / "two_entries.xml").read_text(encoding="utf-8")
        respx.get(API_URL).mock(return_value=httpx.Response(200, text=body))

        page = await client.fetch_page(query)

        assert len(page.papers) == 2
        assert page.total_results == 137

    @respx.mock
    async def test_sends_the_expected_query_parameters(
        self, client: ArxivClient, query: ArxivQuery
    ) -> None:
        route = respx.get(API_URL).mock(
            return_value=httpx.Response(200, text=_feed("", total=0, start=0, per_page=0))
        )

        await client.fetch_page(query, start=40, page_size=20)

        sent = route.calls.last.request.url.params
        assert sent["search_query"] == "(cat:cs.CL)"
        assert sent["start"] == "40"
        assert sent["max_results"] == "20"
        assert sent["sortBy"] == "lastUpdatedDate"

    @respx.mock
    async def test_page_size_is_capped_at_the_api_maximum(
        self, client: ArxivClient, query: ArxivQuery
    ) -> None:
        """Larger pages are rejected by arXiv outright."""
        route = respx.get(API_URL).mock(
            return_value=httpx.Response(200, text=_feed("", total=0, start=0, per_page=0))
        )

        await client.fetch_page(query, page_size=99_999)

        assert route.calls.last.request.url.params["max_results"] == "2000"

    @respx.mock
    async def test_sends_an_identifying_user_agent(
        self, client: ArxivClient, query: ArxivQuery
    ) -> None:
        """A generic user agent is the first thing a service blocks."""
        route = respx.get(API_URL).mock(
            return_value=httpx.Response(200, text=_feed("", total=0, start=0, per_page=0))
        )

        await client.fetch_page(query)

        assert "ai-research-paper-curator" in route.calls.last.request.headers["user-agent"]


@pytest.mark.unit
class TestErrorHandling:
    @respx.mock
    async def test_http_error_becomes_an_arxiv_api_error(
        self, client: ArxivClient, query: ArxivQuery
    ) -> None:
        respx.get(API_URL).mock(return_value=httpx.Response(500))

        with pytest.raises(ArxivApiError) as exc_info:
            await client.fetch_page(query)

        assert exc_info.value.status_code == 500
        assert exc_info.value.service == "arxiv"

    @respx.mock
    async def test_timeout_becomes_an_arxiv_api_error(
        self, client: ArxivClient, query: ArxivQuery
    ) -> None:
        respx.get(API_URL).mock(side_effect=httpx.ConnectTimeout("too slow"))

        with pytest.raises(ArxivApiError, match="timed out"):
            await client.fetch_page(query)

    @respx.mock
    async def test_a_client_error_is_not_retried(self, query: ArxivQuery) -> None:
        """Reissuing a 400 only repeats a request the service already rejected."""
        route = respx.get(API_URL).mock(return_value=httpx.Response(400))
        client = ArxivClient(
            _settings(max_retries=3),
            http_client=httpx.AsyncClient(),
            rate_limiter=AsyncRateLimiter(0),
        )

        with pytest.raises(ArxivApiError) as exc_info:
            await client.fetch_page(query)

        assert exc_info.value.status_code == 400
        assert route.call_count == 1

    @respx.mock
    async def test_a_server_error_is_retried_then_succeeds(self, query: ArxivQuery) -> None:
        route = respx.get(API_URL).mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(200, text=_feed(_entry("2401.00001"), total=1, start=0, per_page=1)),
            ]
        )
        client = ArxivClient(
            _settings(max_retries=1),
            http_client=httpx.AsyncClient(),
            rate_limiter=AsyncRateLimiter(0),
        )

        page = await client.fetch_page(query)

        assert len(page.papers) == 1
        assert route.call_count == 2

    @pytest.mark.parametrize(
        ("status", "retryable"),
        [(400, False), (404, False), (429, True), (500, True), (503, True)],
    )
    def test_retryability_is_decided_by_status_code(self, status: int, retryable: bool) -> None:
        assert ArxivApiError("x", status_code=status).is_retryable is retryable

    def test_a_failure_below_http_is_retryable(self) -> None:
        """No status code means a timeout or reset -- the most retryable case."""
        assert ArxivApiError("connection reset").is_retryable is True


@pytest.mark.unit
class TestPagination:
    @respx.mock
    async def test_pages_until_results_are_exhausted(self, query: ArxivQuery) -> None:
        respx.get(API_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    text=_feed(
                        _entry("2401.00001") + _entry("2401.00002"), total=3, start=0, per_page=2
                    ),
                ),
                httpx.Response(200, text=_feed(_entry("2401.00003"), total=3, start=2, per_page=1)),
            ]
        )
        client = ArxivClient(
            _settings(page_size=2),
            http_client=httpx.AsyncClient(),
            rate_limiter=AsyncRateLimiter(0),
        )

        papers = [paper async for paper in client.iter_papers(query)]

        assert [p.arxiv_id for p in papers] == ["2401.00001", "2401.00002", "2401.00003"]

    @respx.mock
    async def test_limit_stops_iteration_early(self, query: ArxivQuery) -> None:
        """Bounding the work is what keeps a backfill from running for hours."""
        route = respx.get(API_URL).mock(
            return_value=httpx.Response(
                200,
                text=_feed(
                    _entry("2401.00001") + _entry("2401.00002"), total=5000, start=0, per_page=2
                ),
            )
        )
        client = ArxivClient(
            _settings(page_size=2),
            http_client=httpx.AsyncClient(),
            rate_limiter=AsyncRateLimiter(0),
        )

        papers = [paper async for paper in client.iter_papers(query, limit=2)]

        assert len(papers) == 2
        assert route.call_count == 1

    @respx.mock
    async def test_stops_on_an_empty_page_before_the_end(self, query: ArxivQuery) -> None:
        """An empty page can arrive from arXiv mid-result-set.

        Advancing past it would silently skip records, so iteration stops and
        the caller resumes from its checkpoint on the next run.
        """
        respx.get(API_URL).mock(
            side_effect=[
                httpx.Response(
                    200, text=_feed(_entry("2401.00001"), total=100, start=0, per_page=1)
                ),
                httpx.Response(200, text=_feed("", total=100, start=1, per_page=0)),
            ]
        )
        client = ArxivClient(
            _settings(page_size=1),
            http_client=httpx.AsyncClient(),
            rate_limiter=AsyncRateLimiter(0),
        )

        papers = [paper async for paper in client.iter_papers(query)]

        assert len(papers) == 1

    async def test_rejects_a_nonsensical_limit(
        self, client: ArxivClient, query: ArxivQuery
    ) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            [paper async for paper in client.iter_papers(query, limit=0)]


@pytest.mark.unit
class TestFetchByIds:
    @respx.mock
    async def test_requests_the_given_identifiers(self, client: ArxivClient) -> None:
        route = respx.get(API_URL).mock(
            return_value=httpx.Response(
                200,
                text=_feed(
                    _entry("2401.00001") + _entry("2401.00002"), total=2, start=0, per_page=2
                ),
            )
        )

        papers = await client.fetch_by_ids(["2401.00001", "2401.00002"])

        assert len(papers) == 2
        assert route.calls.last.request.url.params["id_list"] == "2401.00001,2401.00002"

    async def test_an_empty_request_makes_no_call(self, client: ArxivClient) -> None:
        assert await client.fetch_by_ids([]) == []


@pytest.mark.unit
class TestLifecycle:
    async def test_does_not_close_an_injected_http_client(self, query: ArxivQuery) -> None:
        """Closing a caller's client would be a surprising side effect."""
        injected = httpx.AsyncClient()
        client = ArxivClient(_settings(), http_client=injected, rate_limiter=AsyncRateLimiter(0))

        await client.aclose()

        assert not injected.is_closed
        await injected.aclose()

    async def test_closes_a_client_it_created(self) -> None:
        async with ArxivClient(_settings(), rate_limiter=AsyncRateLimiter(0)) as client:
            owned = client._http
        assert owned.is_closed
