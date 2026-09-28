"""Optical character recognition fallback, via Tesseract.

Used when a PDF has no usable text layer — a scanned paper. Without this, such
documents index as empty and then never appear in search results, which looks
like a retrieval bug rather than the ingestion problem it is.

Two properties of OCR shape how this module is written.

**It needs a binary that may not be installed.** ``pytesseract`` is a thin
wrapper around the ``tesseract`` executable. The Python package installing
successfully says nothing about whether the binary exists, so availability is
checked explicitly and reported as its own error type. In this project Tesseract
lives in the ingestion container rather than on the developer's machine, so
"unavailable" is the normal local state, not a failure.

**It is slow and CPU-bound.** Seconds per page. Every function here is
synchronous and blocking by design; the calling service runs them on a worker
thread so a thirty-second OCR does not stall the event loop and every other
concurrent request with it.
"""

from __future__ import annotations

import io
import shutil
from functools import lru_cache

import pytesseract
from PIL import Image

from paper_curator.core.logging import get_logger
from paper_curator.ingestion.extraction.errors import OcrUnavailableError

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def is_tesseract_available() -> bool:
    """Return True when the Tesseract binary can be found and run.

    Cached: this shells out, and the answer cannot change within a process.
    Tests that need to simulate absence should call ``cache_clear()``.
    """
    if shutil.which(pytesseract.pytesseract.tesseract_cmd) is None:
        return False
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def require_tesseract() -> None:
    """Raise if OCR is not usable in this environment.

    Raises:
        OcrUnavailableError: The Tesseract binary is missing or unusable.

    """
    if not is_tesseract_available():
        msg = (
            "Tesseract is not installed or not on PATH. OCR runs inside the "
            "ingestion container; a host installation is optional."
        )
        raise OcrUnavailableError(msg)


def ocr_image(png_bytes: bytes, *, language: str = "eng") -> str:
    """Read text from a single rendered page image.

    Blocking and CPU-bound. Call from a worker thread, not the event loop.

    Raises:
        OcrUnavailableError: Tesseract is missing, or the requested language
            pack is not installed.

    """
    require_tesseract()

    try:
        with Image.open(io.BytesIO(png_bytes)) as image:
            return str(pytesseract.image_to_string(image, lang=language))
    except pytesseract.TesseractNotFoundError as exc:
        msg = f"Tesseract binary disappeared mid-run: {exc}"
        raise OcrUnavailableError(msg) from exc
    except pytesseract.TesseractError as exc:
        # The usual cause is a missing language pack: the binary exists, but
        # tesseract-ocr-<lang> was never installed. Worth distinguishing,
        # because the fix is an apt install rather than a code change.
        msg = f"Tesseract failed (language {language!r} may not be installed): {exc}"
        raise OcrUnavailableError(msg, context={"language": language}) from exc


def ocr_images(images: list[bytes], *, language: str = "eng") -> str:
    """Read text from several page images and join them.

    Pages are joined with a form feed, matching how
    :func:`~paper_curator.ingestion.extraction.pdf.extract_text` marks page
    boundaries — so downstream chunking treats OCR output and native text
    identically.

    A page that yields nothing is kept as an empty entry rather than dropped, so
    page numbering stays aligned with the source document.
    """
    if not images:
        return ""

    require_tesseract()

    pages = [ocr_image(image, language=language) for image in images]
    recognised = sum(1 for page in pages if page.strip())

    logger.info(
        "ocr_completed",
        pages_processed=len(images),
        pages_with_text=recognised,
        language=language,
    )
    return "\f".join(pages)
