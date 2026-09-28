"""Tests for the PDF downloader's safety controls.

Every PDF URL originates in the arXiv Atom feed, which is third-party input. The
controls tested here each block a distinct attack or failure, and the tests are
written so that removing any one of them fails loudly rather than silently
widening what the downloader will fetch.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from paper_curator.core.config import ExtractionSettings, Settings
from paper_curator.ingestion.extraction.downloader import PdfDownloader, validate_pdf_url
from paper_curator.ingestion.extraction.errors import (
    PdfDownloadError,
    PdfTooLargeError,
    UnsafeUrlError,
)

PDF_URL = "https://arxiv.org/pdf/2401.12345v1"
MINIMAL_PDF = b"%PDF-1.4\n% a minimal but plausible body\n%%EOF\n"
ALLOWED = ["arxiv.org", "export.arxiv.org"]


def settings_with(**overrides: object) -> Settings:
    return Settings(extraction=ExtractionSettings(**overrides))  # type: ignore[arg-type]


def downloader(**overrides: object) -> PdfDownloader:
    overrides.setdefault("download_max_retries", 0)
    return PdfDownloader(settings_with(**overrides), http_client=httpx.AsyncClient())


@pytest.mark.unit
class TestUrlValidation:
    """The first line of defence, applied before any request is made."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://arxiv.org/pdf/2401.12345v1",
            "http://arxiv.org/pdf/2401.12345v1",
            "https://export.arxiv.org/pdf/2401.12345v1",
            "https://www.arxiv.org/pdf/x",
        ],
    )
    def test_accepts_allowed_hosts(self, url: str) -> None:
        validate_pdf_url(url, [*ALLOWED, "www.arxiv.org"])

    def test_accepts_a_subdomain_of_an_allowed_host(self) -> None:
        validate_pdf_url("https://mirror.arxiv.org/pdf/x", ["arxiv.org"])

    def test_rejects_a_lookalike_domain(self) -> None:
        """A bare suffix check would accept this.

        "evil-arxiv.org".endswith("arxiv.org") is True, which is why the match
        requires either equality or a leading dot.
        """
        with pytest.raises(UnsafeUrlError, match="not in the allowed"):
            validate_pdf_url("https://evil-arxiv.org/pdf/x", ["arxiv.org"])

    def test_rejects_an_arbitrary_host(self) -> None:
        with pytest.raises(UnsafeUrlError, match="not in the allowed"):
            validate_pdf_url("https://example.com/paper.pdf", ALLOWED)

    def test_rejects_the_cloud_metadata_endpoint(self) -> None:
        """The canonical server-side request forgery target.

        A spoofed feed entry pointing here would otherwise make the downloader
        fetch instance credentials on the attacker's behalf.
        """
        with pytest.raises(UnsafeUrlError):
            validate_pdf_url("http://169.254.169.254/latest/meta-data/", ALLOWED)

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://arxiv.org/pdf/x",
            "gopher://arxiv.org/x",
        ],
    )
    def test_rejects_non_http_schemes(self, url: str) -> None:
        with pytest.raises(UnsafeUrlError, match="non-HTTP"):
            validate_pdf_url(url, ALLOWED)

    def test_rejects_a_url_with_no_host(self) -> None:
        with pytest.raises(UnsafeUrlError, match="no host"):
            validate_pdf_url("https:///pdf/x", ALLOWED)

    def test_host_matching_ignores_case(self) -> None:
        validate_pdf_url("https://ArXiv.ORG/pdf/x", ["arxiv.org"])

    def test_an_empty_allowlist_permits_any_host(self) -> None:
        """Intended for tests and local development only."""
        validate_pdf_url("https://example.com/x.pdf", [])


@pytest.mark.unit
class TestSuccessfulDownload:
    @respx.mock
    async def test_returns_the_pdf_bytes(self) -> None:
        respx.get(PDF_URL).mock(return_value=httpx.Response(200, content=MINIMAL_PDF))
        assert await downloader().download(PDF_URL) == MINIMAL_PDF

    @respx.mock
    async def test_a_url_outside_the_allowlist_is_never_requested(self) -> None:
        """Validation happens before the request, not after the response."""
        route = respx.get("https://example.com/x.pdf").mock(
            return_value=httpx.Response(200, content=MINIMAL_PDF)
        )
        with pytest.raises(UnsafeUrlError):
            await downloader().download("https://example.com/x.pdf")
        assert route.call_count == 0


@pytest.mark.unit
class TestSizeLimits:
    @respx.mock
    async def test_rejects_an_oversized_body(self) -> None:
        """Either control may fire; the file must not come back regardless.

        With an honest Content-Length the cheap early check catches it; the
        streaming counter is exercised by the lying-header test below.
        """
        oversized = MINIMAL_PDF + b"\x00" * 5000
        respx.get(PDF_URL).mock(return_value=httpx.Response(200, content=oversized))

        with pytest.raises(PdfTooLargeError, match=r"declares|exceeded"):
            await downloader(max_pdf_bytes=1000).download(PDF_URL)

    @respx.mock
    async def test_rejects_an_oversized_declaration_before_transferring(self) -> None:
        """A cheap early exit when the server is honest about the size."""
        respx.get(PDF_URL).mock(
            return_value=httpx.Response(
                200, content=MINIMAL_PDF, headers={"content-length": "99999999"}
            )
        )
        with pytest.raises(PdfTooLargeError, match="declares"):
            await downloader(max_pdf_bytes=1000).download(PDF_URL)

    @respx.mock
    async def test_a_lying_content_length_does_not_defeat_the_limit(self) -> None:
        """Content-Length is a claim. The byte counter is the real control."""
        oversized = MINIMAL_PDF + b"\x00" * 5000
        respx.get(PDF_URL).mock(
            return_value=httpx.Response(200, content=oversized, headers={"content-length": "10"})
        )
        with pytest.raises(PdfTooLargeError, match="exceeded"):
            await downloader(max_pdf_bytes=1000).download(PDF_URL)

    def test_size_errors_are_never_retried(self) -> None:
        """The file really is that big; another attempt cannot change that."""
        assert PdfTooLargeError("too big", url=PDF_URL, status_code=413).is_retryable is False


@pytest.mark.unit
class TestContentValidation:
    @respx.mock
    async def test_rejects_content_that_is_not_a_pdf(self) -> None:
        """A response can claim application/pdf and contain anything."""
        respx.get(PDF_URL).mock(
            return_value=httpx.Response(
                200,
                content=b"<html>404 not found</html>",
                headers={"content-type": "application/pdf"},
            )
        )
        with pytest.raises(PdfDownloadError, match="not a PDF"):
            await downloader().download(PDF_URL)

    @respx.mock
    async def test_rejects_an_empty_body(self) -> None:
        respx.get(PDF_URL).mock(return_value=httpx.Response(200, content=b""))
        with pytest.raises(PdfDownloadError, match="empty body"):
            await downloader().download(PDF_URL)


@pytest.mark.unit
class TestErrorHandling:
    @respx.mock
    async def test_http_error_is_reported_with_its_status(self) -> None:
        respx.get(PDF_URL).mock(return_value=httpx.Response(404))
        with pytest.raises(PdfDownloadError) as exc_info:
            await downloader().download(PDF_URL)
        assert exc_info.value.status_code == 404

    @respx.mock
    async def test_timeout_is_reported_clearly(self) -> None:
        respx.get(PDF_URL).mock(side_effect=httpx.ReadTimeout("too slow"))
        with pytest.raises(PdfDownloadError, match="timed out"):
            await downloader().download(PDF_URL)

    @respx.mock
    async def test_a_missing_pdf_is_not_retried(self) -> None:
        """A 404 will still be a 404 next minute."""
        route = respx.get(PDF_URL).mock(return_value=httpx.Response(404))
        with pytest.raises(PdfDownloadError):
            await downloader(download_max_retries=3).download(PDF_URL)
        assert route.call_count == 1

    @respx.mock
    async def test_a_server_error_is_retried(self) -> None:
        route = respx.get(PDF_URL).mock(
            side_effect=[httpx.Response(503), httpx.Response(200, content=MINIMAL_PDF)]
        )
        assert await downloader(download_max_retries=1).download(PDF_URL) == MINIMAL_PDF
        assert route.call_count == 2

    @pytest.mark.parametrize(
        ("status", "retryable"),
        [(403, False), (404, False), (429, True), (500, True), (502, True)],
    )
    def test_retryability_follows_the_status_code(self, status: int, retryable: bool) -> None:
        assert PdfDownloadError("x", status_code=status).is_retryable is retryable


@pytest.mark.unit
class TestLifecycle:
    async def test_does_not_close_an_injected_client(self) -> None:
        injected = httpx.AsyncClient()
        await PdfDownloader(settings_with(), http_client=injected).aclose()
        assert not injected.is_closed
        await injected.aclose()
