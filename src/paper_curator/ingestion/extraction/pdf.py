"""Text extraction from PDF bytes using PyMuPDF.

Everything here works **in memory**. PyMuPDF opens from a byte stream and
renders to a byte buffer, so no temporary file is ever created — which removes
an entire category of problems: temp files left behind by a crashed worker,
permission errors in a container, and paths that behave differently on Windows
and Linux.

The module reports *what it found* rather than judging it. Deciding whether the
extracted text is good enough belongs to
:mod:`paper_curator.ingestion.extraction.quality`, so that judgement can be
tested against plain strings without constructing a PDF.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from paper_curator.core.logging import get_logger
from paper_curator.ingestion.extraction.errors import PdfExtractionError

logger = get_logger(__name__)

# A page with fewer characters than this has, in practice, no usable text layer:
# typically just a page number or a header left behind by a scanner.
_MEANINGFUL_PAGE_CHARS = 50


@dataclass(frozen=True, slots=True)
class PdfExtraction:
    """What was found in a PDF, without any judgement about quality."""

    text: str
    page_count: int
    page_char_counts: list[int]
    metadata_title: str | None

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def pages_with_text(self) -> int:
        """Pages carrying a meaningful amount of extractable text."""
        return sum(1 for count in self.page_char_counts if count >= _MEANINGFUL_PAGE_CHARS)

    @property
    def text_page_ratio(self) -> float:
        """Fraction of pages with a usable text layer.

        The clearest signal that a document is scanned. A born-digital paper is
        near 1.0; a scan is at or near 0.0. A value in between usually means a
        digital paper with scanned figures or appendices bound in.
        """
        if self.page_count == 0:
            return 0.0
        return self.pages_with_text / self.page_count


def _open(data: bytes) -> pymupdf.Document:
    """Open a PDF from bytes, converting parser failures into our error type."""
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        msg = f"could not open PDF: {exc}"
        raise PdfExtractionError(msg, context={"bytes": len(data)}) from exc

    if document.needs_pass:
        document.close()
        msg = "PDF is password protected"
        raise PdfExtractionError(msg)

    return document


def extract_text(data: bytes) -> PdfExtraction:
    """Extract the text layer from a PDF.

    Succeeds even when a PDF contains no text at all — that is a scanned
    document, not an error, and it is the case the OCR fallback exists for. Only
    a PDF that cannot be *parsed* raises.

    Raises:
        PdfExtractionError: The file is corrupt, encrypted, or unparseable.

    """
    document = _open(data)
    try:
        page_texts: list[str] = []
        page_char_counts: list[int] = []

        # Indexed access rather than `for page in document`. Documents are
        # iterable at runtime, but PyMuPDF's bundled annotations do not declare
        # __iter__, so the loop form fails type checking for no real benefit.
        for index in range(document.page_count):
            page_text = document[index].get_text()
            page_texts.append(page_text)
            page_char_counts.append(len(page_text.strip()))

        metadata = document.metadata or {}
        title = metadata.get("title") or None

        # Form feed between pages. It survives whitespace normalisation as a
        # page boundary marker, which the chunker uses to avoid splitting across
        # a page break mid-sentence.
        combined = "\f".join(page_texts)

        # A document where every page is empty joins to nothing but separators.
        # Reporting that as N characters of text would hide a scanned document
        # from the emptiness check downstream, so it is normalised to "".
        if not combined.strip():
            combined = ""

        extraction = PdfExtraction(
            text=combined,
            page_count=document.page_count,
            page_char_counts=page_char_counts,
            metadata_title=title.strip() if title else None,
        )
    finally:
        document.close()

    logger.debug(
        "pdf_text_extracted",
        pages=extraction.page_count,
        chars=extraction.char_count,
        pages_with_text=extraction.pages_with_text,
    )
    return extraction


def render_pages_to_png(data: bytes, *, dpi: int, max_pages: int) -> list[bytes]:
    """Render the first ``max_pages`` pages to PNG images, for OCR.

    Bounded on purpose. Rendering is quadratic in DPI and linear in page count,
    and OCR then costs seconds per page on a CPU — so an unbounded 300-page scan
    could occupy a worker for half an hour. Processing the first N pages of a
    scanned paper captures the title, abstract and introduction, which is the
    part most likely to be retrieved anyway.

    Raises:
        PdfExtractionError: The file cannot be opened or a page cannot render.

    """
    if dpi < 1:
        msg = f"dpi must be positive, got {dpi}"
        raise ValueError(msg)
    if max_pages < 1:
        msg = f"max_pages must be at least 1, got {max_pages}"
        raise ValueError(msg)

    document = _open(data)
    try:
        images: list[bytes] = []
        for index in range(min(document.page_count, max_pages)):
            try:
                pixmap = document[index].get_pixmap(dpi=dpi)
                images.append(pixmap.tobytes("png"))
            except Exception as exc:
                msg = f"could not render page {index} at {dpi} dpi: {exc}"
                raise PdfExtractionError(msg, context={"page": index, "dpi": dpi}) from exc
    finally:
        document.close()

    logger.debug("pdf_pages_rendered", pages=len(images), dpi=dpi)
    return images
