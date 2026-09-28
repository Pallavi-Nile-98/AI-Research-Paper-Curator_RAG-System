"""Tests for the text-quality heuristic that decides when OCR runs.

This is the most consequential judgement in ingestion, and it is wrong in two
directions. Running OCR unnecessarily costs seconds per page and produces worse
text than the PDF already contained. Skipping it when needed indexes an empty
document, which then never appears in search results and looks like a retrieval
bug rather than an ingestion one.

Everything here operates on plain strings -- no PDF is constructed -- which is
exactly why the judgement lives in its own module.
"""

from __future__ import annotations

import pytest

from paper_curator.ingestion.extraction.quality import (
    LOW_TEXT_CHARS_PER_PAGE,
    assess_text_quality,
    should_attempt_ocr,
)

# Roughly 1800 characters of ordinary academic prose: what a healthy page of a
# born-digital paper looks like after extraction.
GOOD_PAGE = (
    "We present a systematic study of hybrid retrieval for scientific question "
    "answering over a corpus of academic papers. Our method combines lexical "
    "scoring with dense vector similarity, and we evaluate the combination "
    "against each component in isolation across six categories of query. The "
    "results indicate that the two approaches fail under different conditions, "
    "and we quantify the overlap between their failure cases. We further show "
    "that a cross encoder applied to the fused candidate pool improves ranking "
    "quality at a measurable cost in latency, and we report both figures. "
) * 3


@pytest.mark.unit
class TestHealthyText:
    def test_normal_prose_scores_well(self) -> None:
        report = assess_text_quality(GOOD_PAGE, page_count=1)
        assert report.score > 0.8
        assert report.reasons == []

    def test_healthy_text_does_not_trigger_ocr(self) -> None:
        report = assess_text_quality(GOOD_PAGE, page_count=1)
        assert should_attempt_ocr(report, threshold=0.5) is False

    def test_signals_are_reported_for_inspection(self) -> None:
        """A bare score is not actionable; the components explain it."""
        report = assess_text_quality(GOOD_PAGE, page_count=1)
        assert report.char_count == len(GOOD_PAGE)
        assert report.word_count > 100
        # Clean prose is almost entirely letters once spaces are excluded.
        assert 0.90 < report.alpha_ratio <= 1.0
        assert 3.0 < report.mean_word_length < 8.0


@pytest.mark.unit
class TestScannedDocuments:
    """A scanned PDF is the case OCR actually fixes."""

    def test_empty_extraction_scores_zero(self) -> None:
        report = assess_text_quality("", page_count=10)
        assert report.score == 0.0
        assert report.is_empty
        assert "no text at all" in report.reasons[0]

    def test_empty_extraction_triggers_ocr(self) -> None:
        report = assess_text_quality("", page_count=10)
        assert should_attempt_ocr(report, threshold=0.5) is True

    def test_sparse_text_across_many_pages_triggers_ocr(self) -> None:
        """A scanner often leaves page numbers and a header behind.

        Non-empty, but nowhere near enough to be the paper's content.
        """
        text = "\f".join(f"Page {n}" for n in range(1, 21))
        report = assess_text_quality(text, page_count=20)
        assert report.chars_per_page < LOW_TEXT_CHARS_PER_PAGE
        assert should_attempt_ocr(report, threshold=0.5) is True
        assert any("scanned" in reason for reason in report.reasons)

    def test_the_same_text_on_one_page_is_not_suspicious(self) -> None:
        """Character count alone is meaningless without page count.

        500 characters is a reasonable one-page abstract and an alarming
        20-page paper, which is why density drives the decision.
        """
        short_but_dense = GOOD_PAGE[:900]
        report = assess_text_quality(short_but_dense, page_count=1)
        assert should_attempt_ocr(report, threshold=0.5) is False


@pytest.mark.unit
class TestGarbledText:
    """Text that is present but wrong -- a different problem from a scan."""

    def test_symbol_soup_is_detected(self) -> None:
        report = assess_text_quality("#$%^&*()_+ " * 200, page_count=1)
        assert report.alpha_ratio < 0.2
        assert any("font encoding" in reason for reason in report.reasons)

    def test_replacement_characters_are_detected(self) -> None:
        text = "�" * 300 + GOOD_PAGE
        report = assess_text_quality(text, page_count=1)
        assert any("decoding failure" in reason for reason in report.reasons)

    def test_collapsed_spacing_is_detected(self) -> None:
        """Some PDFs extract an entire page as one unbroken token."""
        report = assess_text_quality("a" * 2000, page_count=1)
        assert any("collapsed spacing" in reason for reason in report.reasons)

    def test_character_fragmentation_is_detected(self) -> None:
        """Others extract every character separately."""
        report = assess_text_quality(" ".join("abcdefghij" * 200), page_count=1)
        assert any("fragmentation" in reason for reason in report.reasons)

    def test_garbled_but_dense_text_does_not_trigger_ocr(self) -> None:
        """OCR repairs a missing text layer, not a broken font encoding.

        Rendering already-garbled glyphs and reading them back produces the same
        garbage more slowly, so this document is flagged rather than reprocessed.
        """
        report = assess_text_quality("#$%^&*()_+ " * 400, page_count=1)
        assert report.score < 0.5
        assert should_attempt_ocr(report, threshold=0.5) is False


@pytest.mark.unit
class TestReportBehaviour:
    def test_describe_includes_the_reason(self) -> None:
        report = assess_text_quality("", page_count=1)
        assert "score=0.00" in report.describe()
        assert "no text at all" in report.describe()

    def test_describe_says_so_when_nothing_is_wrong(self) -> None:
        assert "no issues detected" in assess_text_quality(GOOD_PAGE, page_count=1).describe()

    def test_page_count_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            assess_text_quality("text", page_count=0)

    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_empty_text_triggers_ocr_at_every_threshold(self, threshold: float) -> None:
        """No threshold should ever conclude that zero text is acceptable."""
        report = assess_text_quality("", page_count=5)
        assert should_attempt_ocr(report, threshold=threshold) is True
