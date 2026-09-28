"""Tests for PDF parsing and the OCR fallback decision.

PDFs are built in memory with PyMuPDF rather than committed as binary fixtures.
That keeps the repository free of binaries, makes every case explicit in the
test that uses it, and means a "scanned" document is genuinely a page with no
text layer rather than something approximated.

Tesseract is not installed on the development machine -- OCR runs inside the
ingestion container -- so the OCR *decision* and the surrounding fallback
behaviour are tested with the OCR call stubbed, while anything requiring the
real binary is marked ``requires_ocr`` and skipped when it is absent.
"""

from __future__ import annotations

from collections.abc import Iterator

import pymupdf
import pytest

from paper_curator.core.config import ExtractionSettings, Settings
from paper_curator.db.enums import ExtractionMethod
from paper_curator.ingestion.extraction import ocr as ocr_module
from paper_curator.ingestion.extraction.errors import OcrUnavailableError, PdfExtractionError
from paper_curator.ingestion.extraction.pdf import extract_text, render_pages_to_png
from paper_curator.ingestion.extraction.service import PdfExtractionService

PROSE = (
    "We present a systematic study of hybrid retrieval for scientific question "
    "answering over academic papers. The method combines lexical scoring with "
    "dense vector similarity, and we evaluate the combination against each "
    "component in isolation across several categories of query. Results show "
    "the approaches fail under different conditions. "
)


def make_text_pdf(pages: int = 1, *, body: str = PROSE) -> bytes:
    """Build a born-digital PDF with a real text layer."""
    document = pymupdf.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), body * 4, fontsize=9)
    data: bytes = document.tobytes()
    document.close()
    return data


def make_scanned_pdf(pages: int = 3) -> bytes:
    """Build a PDF whose pages carry no text layer.

    This is what a scanned paper looks like to a parser: valid pages, zero
    extractable characters. It is the case the OCR fallback exists for.
    """
    document = pymupdf.open()
    for _ in range(pages):
        document.new_page()
    data: bytes = document.tobytes()
    document.close()
    return data


def settings_with(**overrides: object) -> Settings:
    """Build settings with extraction options overridden."""
    return Settings(extraction=ExtractionSettings(**overrides))  # type: ignore[arg-type]


@pytest.fixture
def stub_tesseract_present(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pretend Tesseract is installed, without needing the binary."""
    monkeypatch.setattr(ocr_module, "is_tesseract_available", lambda: True)
    yield


@pytest.mark.unit
class TestPdfParsing:
    def test_extracts_text_from_a_born_digital_pdf(self) -> None:
        extraction = extract_text(make_text_pdf())
        assert "hybrid retrieval" in extraction.text
        assert extraction.page_count == 1
        assert extraction.char_count > 500

    def test_reports_per_page_character_counts(self) -> None:
        extraction = extract_text(make_text_pdf(pages=3))
        assert extraction.page_count == 3
        assert len(extraction.page_char_counts) == 3
        assert all(count > 0 for count in extraction.page_char_counts)

    def test_text_page_ratio_is_one_for_a_digital_pdf(self) -> None:
        assert extract_text(make_text_pdf(pages=2)).text_page_ratio == 1.0

    def test_text_page_ratio_is_zero_for_a_scanned_pdf(self) -> None:
        """The clearest single signal that a document was scanned."""
        extraction = extract_text(make_scanned_pdf(pages=4))
        assert extraction.text_page_ratio == 0.0
        assert extraction.char_count == 0

    def test_pages_are_separated_by_a_form_feed(self) -> None:
        """Downstream chunking uses this to avoid splitting across a page break."""
        assert extract_text(make_text_pdf(pages=3)).text.count("\f") == 2

    def test_a_corrupt_file_raises_rather_than_returning_nothing(self) -> None:
        with pytest.raises(PdfExtractionError, match="could not open PDF"):
            extract_text(b"%PDF-1.4 this is not actually a pdf")

    def test_an_empty_pdf_is_not_an_error(self) -> None:
        """A PDF with no text is a scanned document, not a failure."""
        extraction = extract_text(make_scanned_pdf(pages=1))
        assert extraction.char_count == 0


@pytest.mark.unit
class TestPageRendering:
    def test_renders_pages_to_png(self) -> None:
        images = render_pages_to_png(make_text_pdf(pages=2), dpi=72, max_pages=5)
        assert len(images) == 2
        # PNG magic number.
        assert all(image.startswith(b"\x89PNG") for image in images)

    def test_rendering_is_bounded_by_max_pages(self) -> None:
        """Bounds the cost of one pathological document."""
        images = render_pages_to_png(make_text_pdf(pages=10), dpi=72, max_pages=3)
        assert len(images) == 3

    @pytest.mark.parametrize(("dpi", "max_pages"), [(0, 5), (-1, 5), (72, 0)])
    def test_rejects_nonsensical_parameters(self, dpi: int, max_pages: int) -> None:
        with pytest.raises(ValueError, match="must be"):
            render_pages_to_png(make_text_pdf(), dpi=dpi, max_pages=max_pages)


@pytest.mark.unit
class TestExtractionSucceedsWithoutOcr:
    async def test_good_pdf_uses_the_text_layer(self) -> None:
        async with PdfExtractionService(settings_with()) as service:
            result = await service.extract_from_bytes(make_text_pdf(pages=2))

        assert result.method is ExtractionMethod.PYMUPDF
        assert result.is_usable
        assert "hybrid retrieval" in result.text

    async def test_good_pdf_does_not_attempt_ocr(self) -> None:
        """OCR on a perfectly good text layer costs seconds and yields worse text."""
        async with PdfExtractionService(settings_with()) as service:
            result = await service.extract_from_bytes(make_text_pdf(pages=2))

        assert result.ocr_attempted is False
        assert result.ocr_pages == 0

    async def test_records_how_long_extraction_took(self) -> None:
        async with PdfExtractionService(settings_with()) as service:
            result = await service.extract_from_bytes(make_text_pdf())
        assert result.duration_ms >= 0


@pytest.mark.unit
class TestOcrFallbackActivates:
    async def test_scanned_pdf_triggers_ocr(
        self, monkeypatch: pytest.MonkeyPatch, stub_tesseract_present: None
    ) -> None:
        recognised = "Recovered body text from the scanned page. " * 40
        monkeypatch.setattr(ocr_module, "ocr_images", lambda images, language: recognised)

        async with PdfExtractionService(settings_with()) as service:
            result = await service.extract_from_bytes(make_scanned_pdf(pages=3))

        assert result.ocr_attempted is True
        assert result.method is ExtractionMethod.OCR
        assert "Recovered body text" in result.text
        assert result.ocr_pages == 3

    async def test_ocr_respects_the_page_limit(
        self, monkeypatch: pytest.MonkeyPatch, stub_tesseract_present: None
    ) -> None:
        """OCR costs seconds per page; an unbounded scan would occupy a worker."""
        seen: list[int] = []

        def fake_ocr(images: list[bytes], language: str) -> str:
            seen.append(len(images))
            return "Recovered text from the page. " * 40

        monkeypatch.setattr(ocr_module, "ocr_images", fake_ocr)

        async with PdfExtractionService(settings_with(ocr_max_pages=2)) as service:
            await service.extract_from_bytes(make_scanned_pdf(pages=8))

        assert seen == [2]


@pytest.mark.unit
class TestOcrFallbackDoesNotActivate:
    async def test_disabled_by_configuration(self, stub_tesseract_present: None) -> None:
        async with PdfExtractionService(settings_with(ocr_enabled=False)) as service:
            result = await service.extract_from_bytes(make_scanned_pdf(pages=2))

        assert result.ocr_attempted is False
        assert result.ocr_skipped_reason == "ocr_disabled_by_configuration"
        assert result.method is ExtractionMethod.NONE

    async def test_tesseract_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The normal local state: OCR lives in the container, not on the host.

        The document must still come back, flagged, rather than raising.
        """
        monkeypatch.setattr(ocr_module, "is_tesseract_available", lambda: False)

        async with PdfExtractionService(settings_with()) as service:
            result = await service.extract_from_bytes(make_scanned_pdf(pages=2))

        assert result.ocr_attempted is False
        assert result.ocr_skipped_reason == "tesseract_not_installed"
        assert result.is_usable is False

    async def test_ocr_result_is_rejected_when_it_is_worse(
        self, monkeypatch: pytest.MonkeyPatch, stub_tesseract_present: None
    ) -> None:
        """OCR output is not automatically an improvement.

        Misread characters, lost equations and mangled tables can leave it worse
        than a sparse but correct text layer. Keeping the worse result purely
        because the fallback ran would degrade ingestion, not improve it.
        """
        monkeypatch.setattr(ocr_module, "ocr_images", lambda images, language: "### ### ###")

        sparse = make_text_pdf(pages=6, body="Short. ")
        async with PdfExtractionService(settings_with()) as service:
            result = await service.extract_from_bytes(sparse)

        assert result.ocr_attempted is True
        assert result.method is not ExtractionMethod.OCR
        assert "###" not in result.text

    async def test_ocr_failure_does_not_lose_existing_text(
        self, monkeypatch: pytest.MonkeyPatch, stub_tesseract_present: None
    ) -> None:
        """A failed fallback must not discard what was already extracted."""

        def explode(images: list[bytes], language: str) -> str:
            raise OcrUnavailableError("language pack missing")

        monkeypatch.setattr(ocr_module, "ocr_images", explode)

        sparse = make_text_pdf(pages=6, body="Short. ")
        async with PdfExtractionService(settings_with()) as service:
            result = await service.extract_from_bytes(sparse)

        assert result.ocr_attempted is True
        assert result.ocr_skipped_reason == "tesseract_unavailable"
        assert "Short." in result.text


@pytest.mark.unit
class TestNearEmptyExtraction:
    async def test_completely_empty_extraction_is_marked_unusable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ocr_module, "is_tesseract_available", lambda: False)

        async with PdfExtractionService(settings_with()) as service:
            result = await service.extract_from_bytes(make_scanned_pdf(pages=5))

        assert result.char_count == 0
        assert result.method is ExtractionMethod.NONE
        assert result.is_usable is False
        assert result.quality.is_empty

    async def test_empty_extraction_still_reports_the_page_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Needed to record the failure against the right document."""
        monkeypatch.setattr(ocr_module, "is_tesseract_available", lambda: False)

        async with PdfExtractionService(settings_with()) as service:
            result = await service.extract_from_bytes(make_scanned_pdf(pages=7))

        assert result.page_count == 7


@pytest.mark.requires_ocr
@pytest.mark.unit
class TestRealOcr:
    """Exercises the actual Tesseract binary. Skipped when it is not installed."""

    def setup_method(self) -> None:
        ocr_module.is_tesseract_available.cache_clear()
        if not ocr_module.is_tesseract_available():
            pytest.skip("tesseract is not installed on this machine")

    async def test_reads_text_from_a_rendered_page(self) -> None:
        pdf = make_text_pdf(body="The quick brown fox jumps over the lazy dog. ")
        images = render_pages_to_png(pdf, dpi=200, max_pages=1)
        text = ocr_module.ocr_images(images, language="eng")
        assert "quick brown fox" in text.lower()
