"""Heuristics for judging whether extracted PDF text is usable.

This module answers one question: *is the text we just pulled out of a PDF good
enough to index, or should we fall back to OCR?*

Getting that decision wrong is expensive in both directions. Running OCR
unnecessarily costs seconds per page on a CPU and produces worse text than the
PDF already contained. Skipping OCR when it was needed silently indexes an empty
or garbled document, which then never appears in search results and looks like a
retrieval bug rather than an ingestion one.

The signals are deliberately simple and separately testable rather than a single
opaque score. Each one catches a different real failure:

* **Characters per page** — a scanned paper has a text layer of almost nothing.
  This is the strongest signal and catches the common case outright.
* **Alphabetic ratio** — a broken font encoding extracts as symbols and
  punctuation. The text is non-empty but meaningless.
* **Mean word length** — collapsed spacing produces one enormous "word";
  character-level fragmentation produces thousands of single letters. Real prose
  sits between about 3 and 12.
* **Replacement characters** — U+FFFD marks bytes the decoder could not map, a
  direct indicator of encoding failure.

A composite score in [0, 1] combines them, but every component is reported so a
failure can be explained rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

# Below this, a page is almost certainly a scanned image. A dense academic page
# holds roughly 2000-4000 characters; even a sparse title page exceeds 200.
LOW_TEXT_CHARS_PER_PAGE: Final = 100.0

# Above this, characters-per-page stops being evidence of a problem.
GOOD_TEXT_CHARS_PER_PAGE: Final = 800.0

# English academic prose runs roughly 0.70-0.80 alphabetic among non-space
# characters. Much below suggests symbol soup from a broken encoding.
MIN_ALPHA_RATIO: Final = 0.55
GOOD_ALPHA_RATIO: Final = 0.70

MIN_MEAN_WORD_LENGTH: Final = 2.5
MAX_MEAN_WORD_LENGTH: Final = 12.0

# Any measurable quantity of U+FFFD indicates decoding failure.
MAX_REPLACEMENT_RATIO: Final = 0.01

# Multiplier applied when decoding failed. Heavy but not annihilating: some of
# the text is probably still correct, and partial text beats none.
ENCODING_FAILURE_PENALTY: Final = 0.25

REPLACEMENT_CHAR: Final = "�"

_WORD_PATTERN: Final = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Measured signals about a block of extracted text.

    ``reasons`` exists so a low score is actionable. "quality 0.21" tells an
    operator nothing; "0.21 (18 chars/page, looks like a scanned document)" tells
    them the PDF has no text layer and OCR is the right answer.
    """

    score: float
    char_count: int
    page_count: int
    chars_per_page: float
    alpha_ratio: float
    mean_word_length: float
    replacement_char_ratio: float
    word_count: int
    non_space_char_count: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when extraction produced no actual content.

        Counts non-whitespace characters, not raw length. A scanned five-page
        PDF extracts as four form-feed page separators and nothing else;
        measuring raw length would call that "four characters of text" and let
        an empty document pass as merely poor.
        """
        return self.non_space_char_count == 0

    def describe(self) -> str:
        """One-line summary for logs."""
        detail = "; ".join(self.reasons) if self.reasons else "no issues detected"
        return f"score={self.score:.2f} ({detail})"


def _scale(value: float, low: float, high: float) -> float:
    """Map ``value`` onto [0, 1], clamped, rising from ``low`` to ``high``."""
    if high <= low:
        return 1.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def assess_text_quality(text: str, page_count: int) -> QualityReport:
    """Score extracted text and explain the score.

    Args:
        text: Raw extracted text, before any cleaning.
        page_count: Pages in the source PDF. Used to normalise character counts,
            since 500 characters is fine for a one-page abstract and alarming
            for a 20-page paper.

    Returns:
        A :class:`QualityReport` with a score in [0, 1] and the signals behind it.

    """
    if page_count < 1:
        msg = f"page_count must be at least 1, got {page_count}"
        raise ValueError(msg)

    char_count = len(text)
    reasons: list[str] = []

    non_space = [c for c in text if not c.isspace()]

    # Emptiness is measured in non-whitespace characters. A scanned five-page
    # PDF extracts as four form-feed page separators; treating that as content
    # would let a document with nothing in it look merely poor rather than
    # empty, and OCR would never be triggered for it.
    if not non_space:
        return QualityReport(
            score=0.0,
            char_count=char_count,
            page_count=page_count,
            chars_per_page=0.0,
            alpha_ratio=0.0,
            mean_word_length=0.0,
            replacement_char_ratio=0.0,
            word_count=0,
            non_space_char_count=0,
            reasons=["extraction produced no text at all"],
        )

    # Density is measured in real characters too, so a page padded with
    # whitespace does not look denser than one carrying actual words.
    chars_per_page = len(non_space) / page_count

    alpha_ratio = sum(1 for c in non_space if c.isalpha()) / len(non_space)

    words = _WORD_PATTERN.findall(text)
    word_count = len(words)
    mean_word_length = (sum(len(w) for w in words) / word_count) if word_count else 0.0

    replacement_char_ratio = text.count(REPLACEMENT_CHAR) / char_count

    # --- Component scores, each in [0, 1] ---------------------------------
    density_score = _scale(chars_per_page, LOW_TEXT_CHARS_PER_PAGE, GOOD_TEXT_CHARS_PER_PAGE)
    alpha_score = _scale(alpha_ratio, MIN_ALPHA_RATIO, GOOD_ALPHA_RATIO)
    word_shape_score = (
        1.0 if MIN_MEAN_WORD_LENGTH <= mean_word_length <= MAX_MEAN_WORD_LENGTH else 0.0
    )
    encoding_score = (
        ENCODING_FAILURE_PENALTY if replacement_char_ratio > MAX_REPLACEMENT_RATIO else 1.0
    )

    # Legibility MULTIPLIES rather than contributes.
    #
    # An earlier version summed all four components with weights, and a page of
    # pure punctuation scored 0.75: it was dense, and its "words" happened to be
    # a plausible length, which together outweighed having no letters at all.
    # That is backwards. Density only means something once the characters are
    # letters -- four thousand symbols per page is not four thousand characters
    # of text.
    #
    # So legibility gates the rest: no alphabetic content produces a zero score
    # however much of it there is.
    legibility = alpha_score * encoding_score

    # Given the text IS legible, how much of it is there, and is it shaped like
    # prose? Density dominates, because a missing text layer is the failure OCR
    # can actually repair.
    structure = 0.6 * density_score + 0.4 * word_shape_score

    score = legibility * structure

    # --- Explanations -----------------------------------------------------
    if chars_per_page < LOW_TEXT_CHARS_PER_PAGE:
        reasons.append(f"only {chars_per_page:.0f} chars/page, looks like a scanned document")
    if alpha_ratio < MIN_ALPHA_RATIO:
        reasons.append(f"alphabetic ratio {alpha_ratio:.2f} suggests a broken font encoding")
    if word_count and mean_word_length < MIN_MEAN_WORD_LENGTH:
        reasons.append(
            f"mean word length {mean_word_length:.1f} suggests character-level fragmentation"
        )
    if mean_word_length > MAX_MEAN_WORD_LENGTH:
        reasons.append(f"mean word length {mean_word_length:.1f} suggests collapsed spacing")
    if replacement_char_ratio > MAX_REPLACEMENT_RATIO:
        reasons.append(
            f"{replacement_char_ratio:.1%} replacement characters indicate decoding failure"
        )

    return QualityReport(
        score=round(score, 4),
        char_count=char_count,
        page_count=page_count,
        chars_per_page=round(chars_per_page, 2),
        alpha_ratio=round(alpha_ratio, 4),
        mean_word_length=round(mean_word_length, 2),
        replacement_char_ratio=round(replacement_char_ratio, 6),
        word_count=word_count,
        non_space_char_count=len(non_space),
        reasons=reasons,
    )


def should_attempt_ocr(report: QualityReport, *, threshold: float) -> bool:
    """Decide whether OCR is worth running on this document.

    OCR helps exactly one problem: a PDF whose pages are images with no text
    layer. It does not repair a broken font encoding — rendering already-garbled
    glyphs and reading them back produces the same garbage more slowly.

    So the answer is yes when the text is empty or sparse, and no when the text
    is *present but poor*: that document is flagged for review rather than
    burning CPU on a fallback that cannot help it.
    """
    if report.is_empty:
        return True
    if report.chars_per_page < LOW_TEXT_CHARS_PER_PAGE:
        return True
    # Present but low quality: a different problem, and not one OCR solves.
    return report.score < threshold and report.chars_per_page < GOOD_TEXT_CHARS_PER_PAGE
