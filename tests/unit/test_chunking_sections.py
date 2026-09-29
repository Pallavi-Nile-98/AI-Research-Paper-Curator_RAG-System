"""Tests for recovering section structure from flat extracted text.

A PDF has no headings -- only glyphs at coordinates -- so structure is inferred,
and inference can be wrong in two directions. A missed heading costs metadata.
A false positive splits a paragraph in half and produces two incoherent chunks,
which is the worse outcome, so the false-positive cases here matter most.
"""

from __future__ import annotations

import pytest

from paper_curator.ingestion.chunking.sections import (
    SectionKind,
    detect_heading,
    split_into_sections,
)


@pytest.mark.unit
class TestNumberedHeadings:
    @pytest.mark.parametrize(
        ("line", "expected_text", "expected_level"),
        [
            ("1 Introduction", "Introduction", 1),
            ("1. Introduction", "Introduction", 1),
            ("2 Related Work", "Related Work", 1),
            ("3.1 Training Details", "Training Details", 2),
            ("3.1.2 Optimiser Settings", "Optimiser Settings", 3),
            ("10 Conclusion", "Conclusion", 1),
        ],
    )
    def test_numbering_gives_text_and_depth(
        self, line: str, expected_text: str, expected_level: int
    ) -> None:
        """Numbering is the strongest signal, and yields nesting depth free."""
        heading = detect_heading(line)
        assert heading is not None
        assert heading.text == expected_text
        assert heading.level == expected_level

    def test_roman_numerals(self) -> None:
        """IEEE-style papers number sections this way."""
        heading = detect_heading("IV. RESULTS")
        assert heading is not None
        assert heading.number == "IV"
        assert heading.level == 1


@pytest.mark.unit
class TestCanonicalHeadings:
    @pytest.mark.parametrize(
        ("line", "kind"),
        [
            ("Abstract", SectionKind.ABSTRACT),
            ("References", SectionKind.REFERENCES),
            ("Bibliography", SectionKind.REFERENCES),
            ("Acknowledgements", SectionKind.ACKNOWLEDGEMENTS),
            ("Appendix", SectionKind.APPENDIX),
            ("Supplementary Material", SectionKind.APPENDIX),
            ("Introduction", SectionKind.BODY),
            ("Related Work", SectionKind.BODY),
            ("Conclusion", SectionKind.BODY),
        ],
    )
    def test_recognised_without_numbering(self, line: str, kind: SectionKind) -> None:
        heading = detect_heading(line)
        assert heading is not None
        assert heading.kind is kind

    def test_kind_survives_numbering(self) -> None:
        """A numbered References heading is still a reference list."""
        heading = detect_heading("5 References")
        assert heading is not None
        assert heading.kind is SectionKind.REFERENCES

    def test_lettered_appendix_keeps_its_letter(self) -> None:
        heading = detect_heading("Appendix A: Proofs")
        assert heading is not None
        assert heading.kind is SectionKind.APPENDIX
        assert heading.number == "A"

    def test_all_caps_line_is_a_heading(self) -> None:
        heading = detect_heading("EXPERIMENTAL SETUP")
        assert heading is not None
        assert heading.text == "Experimental Setup"


@pytest.mark.unit
class TestFalsePositives:
    """The costly direction: a wrong split breaks a paragraph in two."""

    @pytest.mark.parametrize(
        "line",
        [
            # Body text that happens to begin with a number.
            "2. We then applied the reranker to the candidate pool and measured.",
            "1. The first of several observations we make about this behaviour.",
            # Ordinary sentences.
            "This is an ordinary sentence of body text in a paragraph.",
            "The model was trained for twelve hours on a single GPU.",
            # Too short or too long to be a heading.
            "a",
            "x" * 200,
            "",
            "   ",
        ],
    )
    def test_prose_is_not_treated_as_a_heading(self, line: str) -> None:
        assert detect_heading(line) is None

    def test_a_long_numbered_clause_is_not_a_heading(self) -> None:
        """Length and terminal punctuation both mark it as prose."""
        line = "3.1 of the participants reported that the interface was confusing."
        assert detect_heading(line) is None

    def test_a_lowercase_numbered_line_is_not_a_heading(self) -> None:
        assert detect_heading("1 the quick brown fox") is None


PAPER = """Hybrid Retrieval for Scientific Question Answering

Jane Doe, Rahul Mehta

Abstract

We study the combination of lexical and dense retrieval methods.

1 Introduction

Keeping current with research is genuinely difficult these days.

2 Method

2.1 Retrieval

We combine BM25 with dense vector similarity over identical documents.

2.2 Re-ranking

A cross encoder rescores the fused candidate pool before selection.

3 Results

Hybrid retrieval improved recall at ten over either component alone.

References

[1] Someone et al. A paper about things. 2023.
"""


@pytest.mark.unit
class TestSplittingIntoSections:
    def test_finds_every_section(self) -> None:
        headings = [s.heading for s in split_into_sections(PAPER)]
        assert headings == [
            None,  # title block
            "Abstract",
            "Introduction",
            "Method",
            "Retrieval",
            "Re-ranking",
            "Results",
            "References",
        ]

    def test_text_before_the_first_heading_is_kept(self) -> None:
        """That is the title and author block, which is worth indexing."""
        first = split_into_sections(PAPER)[0]
        assert first.heading is None
        assert first.kind is SectionKind.TITLE
        assert "Hybrid Retrieval" in first.text
        assert "Jane Doe" in first.text

    def test_subsections_record_their_parent(self) -> None:
        """Needed so a chunk can be cited as "Method > Retrieval"."""
        sections = {s.heading: s for s in split_into_sections(PAPER)}
        assert sections["Retrieval"].level == 2
        assert sections["Retrieval"].parent_heading == "Method"
        assert sections["Re-ranking"].parent_heading == "Method"

    def test_a_new_top_level_section_clears_the_parent(self) -> None:
        sections = {s.heading: s for s in split_into_sections(PAPER)}
        assert sections["Results"].parent_heading is None

    def test_section_text_excludes_its_own_heading(self) -> None:
        sections = {s.heading: s for s in split_into_sections(PAPER)}
        assert not sections["Introduction"].text.startswith("1 Introduction")
        assert "Keeping current" in sections["Introduction"].text

    def test_page_separators_do_not_become_content(self) -> None:
        text = "Abstract\n\nSome abstract text here.\f1 Introduction\n\nBody text."
        headings = [s.heading for s in split_into_sections(text)]
        assert "Introduction" in headings

    def test_a_paper_with_no_recognised_headings_still_returns_one_section(self) -> None:
        """Detection failing must not mean losing the paper."""
        text = "just some text\n\nwith two paragraphs and nothing resembling a heading"
        sections = split_into_sections(text)
        assert len(sections) == 1
        assert sections[0].heading is None
        assert "two paragraphs" in sections[0].text

    def test_empty_input_produces_no_sections(self) -> None:
        assert split_into_sections("") == []
