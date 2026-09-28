"""Orchestrates PDF download, text extraction and the OCR fallback.

The decision this module makes on every document:

    extract the text layer
        -> is it good enough?
            yes -> done
            no  -> would OCR help?
                     yes -> OCR, then keep whichever result is better
                     no  -> keep what we have, flagged as poor

Two details are easy to get wrong and are handled explicitly.

**OCR output is not automatically better.** A PDF with a sparse but correct text
layer can produce OCR output that is worse — misread characters, lost equations,
mangled tables. So both results are scored and the better one wins, rather than
OCR being assumed to be an improvement because it ran.

**Blocking work runs on a worker thread.** Parsing, rendering and OCR are all
CPU-bound and synchronous. Awaiting them directly would block the event loop for
the whole duration, stalling every other concurrent request. ``asyncio.to_thread``
keeps the loop responsive.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.logging import get_logger
from paper_curator.db.enums import ExtractionMethod
from paper_curator.ingestion.extraction import ocr as ocr_module
from paper_curator.ingestion.extraction import pdf as pdf_module
from paper_curator.ingestion.extraction.downloader import PdfDownloader
from paper_curator.ingestion.extraction.errors import OcrUnavailableError
from paper_curator.ingestion.extraction.quality import (
    QualityReport,
    assess_text_quality,
    should_attempt_ocr,
)

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Extracted text plus everything needed to explain how it was obtained."""

    text: str
    page_count: int
    method: ExtractionMethod
    quality: QualityReport
    duration_ms: int

    ocr_attempted: bool = False
    ocr_pages: int = 0
    ocr_skipped_reason: str | None = None

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_usable(self) -> bool:
        """True when there is enough text to be worth indexing.

        Deliberately not a quality judgement: imperfect text from a difficult
        scan is still far more useful than nothing.

        Delegates emptiness to the quality report, which counts non-whitespace
        characters. Using raw length here would call a scanned document's page
        separators "content".
        """
        return self.method is not ExtractionMethod.NONE and not self.quality.is_empty


class PdfExtractionService:
    """Turns a PDF URL or PDF bytes into usable text."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        downloader: PdfDownloader | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._cfg = self._settings.extraction
        self._owns_downloader = downloader is None
        self._downloader = downloader or PdfDownloader(self._settings)

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
        """Release the downloader's connection pool, if we own it."""
        if self._owns_downloader:
            await self._downloader.aclose()

    async def extract_from_url(self, url: str) -> ExtractionResult:
        """Download a PDF and extract its text."""
        data = await self._downloader.download(url)
        return await self.extract_from_bytes(data)

    async def extract_from_bytes(self, data: bytes) -> ExtractionResult:
        """Extract text from PDF bytes, falling back to OCR when warranted.

        Raises:
            PdfExtractionError: The PDF cannot be parsed at all.

        """
        started = time.perf_counter()

        # Parsing is synchronous and CPU-bound; keep it off the event loop.
        extraction = await asyncio.to_thread(pdf_module.extract_text, data)
        quality = assess_text_quality(extraction.text, extraction.page_count)

        logger.info(
            "pdf_extracted",
            pages=extraction.page_count,
            chars=extraction.char_count,
            pages_with_text=extraction.pages_with_text,
            text_page_ratio=round(extraction.text_page_ratio, 2),
            quality=quality.describe(),
        )

        wants_ocr = should_attempt_ocr(quality, threshold=self._cfg.quality_threshold)
        if not wants_ocr:
            return self._result(
                text=extraction.text,
                page_count=extraction.page_count,
                method=ExtractionMethod.PYMUPDF,
                quality=quality,
                started=started,
            )

        skip_reason = self._ocr_skip_reason()
        if skip_reason is not None:
            logger.warning(
                "ocr_skipped",
                reason=skip_reason,
                quality=quality.describe(),
                detail="document will be indexed with poor or missing text",
            )
            return self._result(
                text=extraction.text,
                page_count=extraction.page_count,
                method=(ExtractionMethod.NONE if quality.is_empty else ExtractionMethod.PYMUPDF),
                quality=quality,
                started=started,
                ocr_skipped_reason=skip_reason,
            )

        return await self._run_ocr_fallback(
            data,
            native_text=extraction.text,
            native_quality=quality,
            page_count=extraction.page_count,
            started=started,
        )

    def _ocr_skip_reason(self) -> str | None:
        """Return why OCR will not run, or None when it can."""
        if not self._cfg.ocr_enabled:
            return "ocr_disabled_by_configuration"
        if not ocr_module.is_tesseract_available():
            return "tesseract_not_installed"
        return None

    async def _run_ocr_fallback(
        self,
        data: bytes,
        *,
        native_text: str,
        native_quality: QualityReport,
        page_count: int,
        started: float,
    ) -> ExtractionResult:
        """Render pages, OCR them, and keep whichever text scores higher."""
        logger.info(
            "ocr_fallback_triggered",
            reason=native_quality.describe(),
            max_pages=self._cfg.ocr_max_pages,
            dpi=self._cfg.ocr_dpi,
        )

        try:
            images = await asyncio.to_thread(
                pdf_module.render_pages_to_png,
                data,
                dpi=self._cfg.ocr_dpi,
                max_pages=self._cfg.ocr_max_pages,
            )
            ocr_text = await asyncio.wait_for(
                asyncio.to_thread(ocr_module.ocr_images, images, language=self._cfg.ocr_language),
                timeout=self._cfg.ocr_timeout_seconds,
            )
        except (TimeoutError, OcrUnavailableError) as exc:
            # A failed fallback must not lose the text we already had.
            reason = "ocr_timed_out" if isinstance(exc, TimeoutError) else "tesseract_unavailable"
            logger.warning("ocr_failed", reason=reason, error=str(exc))
            return self._result(
                text=native_text,
                page_count=page_count,
                method=(
                    ExtractionMethod.NONE if native_quality.is_empty else ExtractionMethod.PYMUPDF
                ),
                quality=native_quality,
                started=started,
                ocr_attempted=True,
                ocr_skipped_reason=reason,
            )

        # OCR ran on at most ocr_max_pages, so score it against the pages it
        # actually saw. Dividing by the full page count would understate its
        # density and could make a good OCR result look like a failure.
        ocr_quality = assess_text_quality(ocr_text, max(1, len(images)))

        if ocr_quality.score > native_quality.score:
            logger.info(
                "ocr_result_accepted",
                ocr_score=ocr_quality.score,
                native_score=native_quality.score,
                ocr_chars=len(ocr_text),
            )
            return self._result(
                text=ocr_text,
                page_count=page_count,
                method=ExtractionMethod.OCR,
                quality=ocr_quality,
                started=started,
                ocr_attempted=True,
                ocr_pages=len(images),
            )

        # OCR did not help. Keeping the worse result purely because the fallback
        # ran would make ingestion quality worse, not better.
        logger.info(
            "ocr_result_rejected",
            ocr_score=ocr_quality.score,
            native_score=native_quality.score,
            detail="native extraction scored higher; keeping it",
        )
        return self._result(
            text=native_text,
            page_count=page_count,
            method=(ExtractionMethod.NONE if native_quality.is_empty else ExtractionMethod.PYMUPDF),
            quality=native_quality,
            started=started,
            ocr_attempted=True,
            ocr_pages=len(images),
        )

    @staticmethod
    def _result(
        *,
        text: str,
        page_count: int,
        method: ExtractionMethod,
        quality: QualityReport,
        started: float,
        ocr_attempted: bool = False,
        ocr_pages: int = 0,
        ocr_skipped_reason: str | None = None,
    ) -> ExtractionResult:
        return ExtractionResult(
            text=text,
            page_count=page_count,
            method=method,
            quality=quality,
            duration_ms=int((time.perf_counter() - started) * 1000),
            ocr_attempted=ocr_attempted,
            ocr_pages=ocr_pages,
            ocr_skipped_reason=ocr_skipped_reason,
        )
