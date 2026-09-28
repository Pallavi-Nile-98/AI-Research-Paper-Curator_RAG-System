"""PDF download, text extraction and OCR fallback.

    from paper_curator.ingestion.extraction import PdfExtractionService

    async with PdfExtractionService() as service:
        result = await service.extract_from_url(paper.pdf_url)
        if result.is_usable:
            ...

The pieces are separable on purpose, so each can be tested in isolation:
``quality`` judges plain strings with no PDF involved, ``downloader`` handles
bytes over HTTP with no parsing, ``pdf`` parses with no network, and ``service``
composes them.
"""

from paper_curator.ingestion.extraction.downloader import PdfDownloader, validate_pdf_url
from paper_curator.ingestion.extraction.errors import (
    OcrUnavailableError,
    PdfDownloadError,
    PdfExtractionError,
    PdfTooLargeError,
    UnsafeUrlError,
)
from paper_curator.ingestion.extraction.ocr import (
    is_tesseract_available,
    ocr_image,
    ocr_images,
)
from paper_curator.ingestion.extraction.pdf import (
    PdfExtraction,
    extract_text,
    render_pages_to_png,
)
from paper_curator.ingestion.extraction.quality import (
    QualityReport,
    assess_text_quality,
    should_attempt_ocr,
)
from paper_curator.ingestion.extraction.service import ExtractionResult, PdfExtractionService

__all__ = [
    "ExtractionResult",
    "OcrUnavailableError",
    "PdfDownloadError",
    "PdfDownloader",
    "PdfExtraction",
    "PdfExtractionError",
    "PdfExtractionService",
    "PdfTooLargeError",
    "QualityReport",
    "UnsafeUrlError",
    "assess_text_quality",
    "extract_text",
    "is_tesseract_available",
    "ocr_image",
    "ocr_images",
    "render_pages_to_png",
    "should_attempt_ocr",
    "validate_pdf_url",
]
