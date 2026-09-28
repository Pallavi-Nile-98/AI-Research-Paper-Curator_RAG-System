"""Download PDFs safely.

PDF URLs arrive from the arXiv Atom feed. That feed is third-party input, so
this module treats every URL as untrusted and applies four independent checks.
Each one blocks a distinct failure that the others do not:

1. **Host allowlist, before any request.** A spoofed or compromised feed could
   otherwise point the downloader at ``http://169.254.169.254/`` — the cloud
   metadata endpoint — or at an internal service reachable only from inside the
   deployment. That is server-side request forgery, and a URL allowlist is the
   cheap defence.

2. **Host re-validation after redirects.** Redirects are followed, because arXiv
   legitimately redirects HTTP to HTTPS. But a redirect is just another URL
   chosen by someone else, so the final destination is checked too. Validating
   only the initial URL would leave the door open.

3. **Size enforced while streaming.** ``Content-Length`` is a claim, not a fact:
   it can be absent, or wrong, or deliberately understated. The limit is
   therefore enforced against bytes actually received, and the connection is
   abandoned the moment it is exceeded — so a hostile endpoint cannot exhaust
   memory by streaming forever.

4. **Magic bytes.** A response can claim ``application/pdf`` and contain
   anything. Every PDF begins with ``%PDF-``; checking is one comparison and
   stops the parser being handed something unexpected.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.exceptions import IngestionError
from paper_curator.core.logging import get_logger
from paper_curator.core.retry import retry_async
from paper_curator.ingestion.extraction.errors import (
    PdfDownloadError,
    PdfTooLargeError,
    UnsafeUrlError,
)

logger = get_logger(__name__)

PDF_MAGIC = b"%PDF-"
ALLOWED_SCHEMES = frozenset({"http", "https"})

# Read in chunks rather than all at once so the size limit can be enforced
# during the transfer instead of after it.
_CHUNK_SIZE = 64 * 1024


def validate_pdf_url(url: str, allowed_hosts: list[str]) -> None:
    """Check a URL is safe to fetch, raising if it is not.

    Args:
        url: The URL to validate.
        allowed_hosts: Permitted hostnames. An empty list disables the check,
            which is intended only for tests and local development.

    Raises:
        UnsafeUrlError: The scheme is not HTTP(S), the URL has no host, or the
            host is not in ``allowed_hosts``.

    """
    parsed = urlparse(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        msg = f"refusing non-HTTP(S) URL scheme {parsed.scheme!r}"
        raise UnsafeUrlError(msg, url=url)

    host = parsed.hostname
    if not host:
        msg = "URL has no host"
        raise UnsafeUrlError(msg, url=url)

    if not allowed_hosts:
        return

    host = host.lower()
    # Exact match or a subdomain of an allowed host. A bare suffix check would
    # accept "evil-arxiv.org" for the suffix "arxiv.org", so the dot matters.
    permitted = any(
        host == allowed.lower() or host.endswith(f".{allowed.lower()}") for allowed in allowed_hosts
    )
    if not permitted:
        msg = f"host {host!r} is not in the allowed PDF hosts"
        raise UnsafeUrlError(msg, url=url, context={"allowed_hosts": allowed_hosts})


class PdfDownloader:
    """Fetches PDF bytes over HTTP with size, host and content checks."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        cfg = self._settings.extraction

        self._max_bytes = cfg.max_pdf_bytes
        self._allowed_hosts = list(cfg.allowed_pdf_hosts)
        self._max_retries = cfg.download_max_retries

        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.download_timeout_seconds),
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        """Close the HTTP client, but only if this instance created it."""
        if self._owns_client:
            await self._http.aclose()

    async def _fetch(self, url: str) -> bytes:
        """Stream one response, enforcing the size limit as bytes arrive."""
        try:
            async with self._http.stream("GET", url) as response:
                # The final URL after redirects is a destination someone else
                # chose. Validate it too.
                validate_pdf_url(str(response.url), self._allowed_hosts)

                if response.status_code != httpx.codes.OK:
                    msg = f"PDF download returned HTTP {response.status_code}"
                    raise PdfDownloadError(msg, url=url, status_code=response.status_code)

                # Advisory only: reject an obviously oversized file before
                # transferring it. The real check is below.
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > self._max_bytes:
                    msg = f"PDF declares {int(declared)} bytes, limit is {self._max_bytes}"
                    raise PdfTooLargeError(msg, url=url)

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes(_CHUNK_SIZE):
                    total += len(chunk)
                    if total > self._max_bytes:
                        # Abandon the connection rather than reading to the end:
                        # an endpoint that streams indefinitely must not be able
                        # to exhaust memory.
                        msg = f"PDF exceeded {self._max_bytes} bytes during download"
                        raise PdfTooLargeError(msg, url=url)
                    chunks.append(chunk)

        except (PdfDownloadError, UnsafeUrlError):
            raise
        except httpx.TimeoutException as exc:
            msg = f"PDF download timed out: {exc}"
            raise PdfDownloadError(msg, url=url) from exc
        except httpx.HTTPError as exc:
            msg = f"PDF download failed: {exc}"
            raise PdfDownloadError(msg, url=url) from exc

        return b"".join(chunks)

    async def download(self, url: str) -> bytes:
        """Download and validate a PDF.

        Raises:
            UnsafeUrlError: The URL is not permitted.
            PdfTooLargeError: The file exceeds the configured size limit.
            PdfDownloadError: The transfer failed, or the content is not a PDF.

        """
        validate_pdf_url(url, self._allowed_hosts)

        async def attempt() -> bytes:
            try:
                return await self._fetch(url)
            except PdfDownloadError as exc:
                if not exc.is_retryable:
                    raise _PermanentDownloadError(exc) from exc
                raise

        try:
            data = await retry_async(
                attempt,
                max_attempts=self._max_retries + 1,
                retry_on=(PdfDownloadError,),
                operation_name="pdf.download",
            )
        except _PermanentDownloadError as wrapper:
            raise wrapper.original from wrapper

        if not data:
            msg = "PDF download returned an empty body"
            raise PdfDownloadError(msg, url=url)

        # A response can claim application/pdf and contain anything at all.
        if not data.startswith(PDF_MAGIC):
            preview = data[:16].hex()
            msg = "downloaded content is not a PDF (missing %PDF- header)"
            raise PdfDownloadError(msg, url=url, context={"first_bytes_hex": preview})

        logger.info("pdf_downloaded", url=url, bytes=len(data))
        return data


class _PermanentDownloadError(IngestionError):
    """Internal wrapper marking a failure the retry policy must not catch.

    Whether a download failure is retryable depends on its status code, but
    ``retry_async`` dispatches on exception *type*. Wrapping non-retryable
    failures in a type outside ``retry_on`` lets a 404 fail immediately while a
    503 is retried.

    It inherits from :class:`IngestionError` rather than from
    :class:`PdfDownloadError`, and that detail is the whole mechanism. An
    earlier version subclassed ``PdfDownloadError`` -- which is precisely what
    ``retry_on`` catches -- so the wrapper intended to escape the retry loop was
    caught by it, and a 404 was reissued four times. Any subclass of a type in
    ``retry_on`` is still retried.
    """

    def __init__(self, original: PdfDownloadError) -> None:
        super().__init__(str(original), context=original.context)
        self.original = original
