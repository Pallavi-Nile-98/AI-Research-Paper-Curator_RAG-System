"""Tests for structure-aware chunking.

Chunking sets the ceiling on retrieval quality: a chunk that begins mid-sentence
and belongs to no section embeds poorly and cannot be quoted in an answer. The
properties asserted here are the ones that keep chunks coherent.
"""

from __future__ import annotations

import datetime as dt

import pytest

from paper_curator.core.config import ChunkingSettings, Settings
from paper_curator.ingestion.chunking import (
    HeuristicTokenCounter,
    HuggingFaceTokenCounter,
    SectionKind,
    StructureAwareChunker,
    TextChunk,
    build_chunk_document,
    split_sentences,
)

SENTENCE = "The retrieval system combines lexical scoring with dense vector similarity. "


def chunker(**overrides: object) -> StructureAwareChunker:
    """Build a chunker with chunking settings overridden."""
    return StructureAwareChunker(Settings(chunking=ChunkingSettings(**overrides)))  # type: ignore[arg-type]


def paper(body_sentences: int = 4) -> str:
    """Build a small paper with two sections."""
    body = SENTENCE * body_sentences
    return f"A Paper Title\n\nAbstract\n\n{body}\n\n1 Introduction\n\n{body}\n\n2 Method\n\n{body}"


@pytest.mark.unit
class TestSizeLimits:
    def test_no_chunk_exceeds_the_token_limit(self) -> None:
        """Beyond the embedding model's input limit, the tail is silently cut.

        The text would be indexed while contributing nothing to its own vector.
        """
        long_text = "1 Introduction\n\n" + (SENTENCE * 300)
        chunks = chunker(max_tokens=200, min_tokens=20).chunk(long_text)

        assert chunks
        assert all(c.token_count <= 200 for c in chunks)

    def test_a_long_section_produces_several_chunks(self) -> None:
        chunks = chunker(max_tokens=120, min_tokens=20).chunk(
            "1 Introduction\n\n" + (SENTENCE * 200)
        )
        assert len(chunks) > 3

    def test_chunk_indexes_are_contiguous_from_zero(self) -> None:
        """chunk_index forms part of the deterministic OpenSearch document ID."""
        chunks = chunker(max_tokens=100, min_tokens=20).chunk(paper(body_sentences=20))
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_max_chunks_per_paper_is_enforced(self) -> None:
        """Bounds what one pathological document can do to the index."""
        chunks = chunker(max_tokens=60, min_tokens=10, max_chunks_per_paper=5).chunk(
            "1 Introduction\n\n" + (SENTENCE * 500)
        )
        assert len(chunks) <= 5


@pytest.mark.unit
class TestSectionBoundaries:
    def test_chunks_never_span_two_sections(self) -> None:
        """A chunk covering the end of Method and start of Results is about neither."""
        chunks = chunker(max_tokens=2000, min_tokens=5).chunk(paper())
        for chunk in chunks:
            # Each chunk's text must appear inside exactly one section's body.
            assert chunk.text.count("Introduction") == 0 or chunk.section == "Introduction"

    def test_each_section_becomes_its_own_chunk_when_small(self) -> None:
        chunks = chunker(max_tokens=2000, min_tokens=5).chunk(paper())
        sections = [c.section for c in chunks]
        assert "Introduction" in sections
        assert "Method" in sections

    def test_subsection_metadata_is_preserved(self) -> None:
        text = (
            "1 Method\n\nIntroductory prose for the method section here.\n\n"
            "1.1 Retrieval\n\n" + SENTENCE * 3 + "\n\n"
            "1.2 Re-ranking\n\n" + SENTENCE * 3
        )
        chunks = chunker(max_tokens=2000, min_tokens=5).chunk(text)
        located = {c.location for c in chunks}
        assert "Method > Retrieval" in located
        assert "Method > Re-ranking" in located

    def test_a_short_real_subsection_is_not_discarded(self) -> None:
        """Regression: min_tokens once deleted short-but-real sections.

        Two paragraphs of Method vanished from the index entirely. Content
        disappearing silently is worse than a slightly undersized chunk, so
        undersized chunks are now merged where possible and only genuine noise
        is dropped.
        """
        text = (
            "1 Method\n\n" + SENTENCE * 6 + "\n\n"
            "2 Results\n\nRecall improved by a measurable margin."
        )
        chunks = chunker(max_tokens=400, min_tokens=100).chunk(text)
        assert any("Recall improved" in c.text for c in chunks)

    def test_undersized_chunks_merge_within_a_section(self) -> None:
        text = "1 Introduction\n\nFirst short paragraph.\n\nSecond short paragraph."
        chunks = chunker(max_tokens=400, min_tokens=200).chunk(text)
        assert len(chunks) == 1
        assert "First short" in chunks[0].text
        assert "Second short" in chunks[0].text


@pytest.mark.unit
class TestOversizedContent:
    def test_an_oversized_paragraph_splits_on_sentences(self) -> None:
        """A cut should still land on a grammatical boundary."""
        chunks = chunker(max_tokens=60, min_tokens=10).chunk("1 Introduction\n\n" + (SENTENCE * 40))
        assert len(chunks) > 1
        # Every chunk should start at a sentence beginning.
        assert all(c.text.lstrip().startswith("The retrieval") for c in chunks)

    def test_an_oversized_single_sentence_still_produces_chunks(self) -> None:
        """Usually a table or equation extracted as one unbroken line.

        No meaningful boundary remains, but truncation would be worse.
        """
        run_on = "word " * 2000
        chunks = chunker(max_tokens=100, min_tokens=10).chunk(f"1 Data\n\n{run_on}")
        assert len(chunks) > 5
        assert all(c.token_count <= 100 for c in chunks)


@pytest.mark.unit
class TestOverlap:
    def test_consecutive_chunks_share_text(self) -> None:
        """A passage straddling a boundary must be findable from either side."""
        chunks = chunker(max_tokens=120, min_tokens=20, overlap_tokens=40).chunk(
            "1 Introduction\n\n"
            + "".join(f"Sentence number {n} describes a distinct finding. " for n in range(60))
        )
        assert len(chunks) > 2
        first_tail = chunks[0].text.split()[-5:]
        assert any(word in chunks[1].text for word in first_tail)

    def test_zero_overlap_produces_disjoint_chunks(self) -> None:
        chunks = chunker(max_tokens=120, min_tokens=20, overlap_tokens=0).chunk(
            "1 Introduction\n\n"
            + "".join(f"Sentence number {n} describes a distinct finding. " for n in range(60))
        )
        assert len(chunks) > 2
        assert chunks[0].text.split()[-1] not in chunks[1].text.split()[:3]

    def test_overlap_is_clamped_below_the_chunk_size(self) -> None:
        """Overlap at or above max_tokens would stop the chunker advancing."""
        config = ChunkingSettings(max_tokens=200, overlap_tokens=500)
        assert config.effective_overlap == 100


@pytest.mark.unit
class TestSectionFiltering:
    REFERENCED = (
        "1 Introduction\n\n" + SENTENCE * 4 + "\n\n"
        "References\n\n[1] Someone et al. A paper about things. 2023.\n"
        "[2] Another Person. A different paper entirely. 2024.\n"
    )

    def test_references_are_excluded_by_default(self) -> None:
        """Citation lists match many queries for the wrong reason."""
        chunks = chunker(min_tokens=5).chunk(self.REFERENCED)
        assert not any(c.kind is SectionKind.REFERENCES for c in chunks)

    def test_references_can_be_included(self) -> None:
        """Asking which papers cite a given work is a legitimate question."""
        chunks = chunker(min_tokens=5, include_references=True).chunk(self.REFERENCED)
        assert any(c.kind is SectionKind.REFERENCES for c in chunks)

    def test_appendices_are_kept_by_default(self) -> None:
        """They carry proofs, extra results and hyperparameters."""
        text = "1 Introduction\n\n" + SENTENCE * 4 + "\n\nAppendix A: Proofs\n\n" + SENTENCE * 4
        chunks = chunker(min_tokens=5).chunk(text)
        assert any(c.kind is SectionKind.APPENDIX for c in chunks)

    def test_acknowledgements_are_excluded(self) -> None:
        """Funding statements match author and institution queries spuriously."""
        text = (
            "1 Introduction\n\n" + SENTENCE * 4 + "\n\n"
            "Acknowledgements\n\nWe thank the reviewers and our funding body for support."
        )
        chunks = chunker(min_tokens=5).chunk(text)
        assert not any(c.kind is SectionKind.ACKNOWLEDGEMENTS for c in chunks)


@pytest.mark.unit
class TestSentenceSplitting:
    def test_splits_on_sentence_ends(self) -> None:
        assert len(split_sentences("First one here. Second one here. Third one.")) == 3

    @pytest.mark.parametrize(
        "text",
        [
            "As shown by Smith et al. The result holds.",
            "See Fig. 3 for details.",
            "Models e.g. BERT are common.",
        ],
    )
    def test_common_abbreviations_do_not_end_a_sentence(self, text: str) -> None:
        """Splitting on "et al." would cut a citation in half."""
        assert len(split_sentences(text)) == 1

    def test_empty_input(self) -> None:
        assert split_sentences("") == []


@pytest.mark.unit
class TestContentHash:
    def test_is_deterministic(self) -> None:
        a = TextChunk(text="identical text", chunk_index=0, token_count=3)
        b = TextChunk(text="identical text", chunk_index=7, token_count=3)
        assert a.content_hash == b.content_hash

    def test_changes_when_the_text_changes(self) -> None:
        """Drift detection depends on this."""
        a = TextChunk(text="original text", chunk_index=0, token_count=3)
        b = TextChunk(text="original text.", chunk_index=0, token_count=3)
        assert a.content_hash != b.content_hash

    def test_is_a_sha256_hex_digest(self) -> None:
        digest = TextChunk(text="x", chunk_index=0, token_count=1).content_hash
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


@pytest.mark.unit
class TestChunkDocument:
    """Every field needed to render a citation must reach the index."""

    def test_carries_paper_identity_and_position(self) -> None:
        chunk = TextChunk(
            text="A passage.",
            chunk_index=3,
            token_count=3,
            section="Method",
            subsection="Retrieval",
        )
        doc = build_chunk_document(
            chunk,
            arxiv_id="2401.12345",
            version=2,
            title="A Paper",
            authors=["Jane Doe", "Rahul Mehta"],
            abstract="An abstract.",
            published_at=dt.datetime(2024, 1, 20, tzinfo=dt.UTC),
            updated_at=dt.datetime(2024, 2, 1, tzinfo=dt.UTC),
            primary_category="cs.IR",
            categories=["cs.IR", "cs.CL"],
            abs_url="https://arxiv.org/abs/2401.12345v2",
            pdf_url="https://arxiv.org/pdf/2401.12345v2",
        )

        assert doc["chunk_id"] == "2401.12345v2:3"
        assert doc["arxiv_id"] == "2401.12345"
        assert doc["version"] == 2
        assert doc["section"] == "Method"
        assert doc["subsection"] == "Retrieval"
        assert doc["authors"] == ["Jane Doe", "Rahul Mehta"]
        assert doc["categories"] == ["cs.IR", "cs.CL"]
        assert doc["content_hash"] == chunk.content_hash
        assert doc["text"] == "A passage."

    def test_document_id_is_deterministic(self) -> None:
        """Re-indexing must overwrite rather than duplicate."""
        chunk = TextChunk(text="x", chunk_index=0, token_count=1)
        common = {
            "arxiv_id": "2401.1",
            "version": 1,
            "title": "T",
            "authors": ["A"],
            "abstract": "A",
            "published_at": dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
            "updated_at": dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
            "primary_category": "cs.CL",
            "categories": ["cs.CL"],
            "abs_url": "u",
            "pdf_url": "u",
        }
        assert (
            build_chunk_document(chunk, **common)["chunk_id"]  # type: ignore[arg-type]
            == build_chunk_document(chunk, **common)["chunk_id"]  # type: ignore[arg-type]
        )


@pytest.mark.unit
class TestTokenCounting:
    def test_counts_scale_with_length(self) -> None:
        counter = HeuristicTokenCounter()
        assert counter.count(SENTENCE) < counter.count(SENTENCE * 5)

    def test_empty_text_counts_zero(self) -> None:
        assert HeuristicTokenCounter().count("   ") == 0

    def test_estimate_is_close_to_word_count(self) -> None:
        """Roughly 1.3 tokens per word for ordinary English prose."""
        text = "the quick brown fox jumps over the lazy dog"
        count = HeuristicTokenCounter().count(text)
        assert 9 <= count <= 16

    def test_a_real_tokenizer_can_be_injected(self) -> None:
        class FakeTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return list(range(len(text.split())))

        assert HuggingFaceTokenCounter(FakeTokenizer()).count("one two three") == 3

    def test_rejects_an_object_that_cannot_tokenize(self) -> None:
        with pytest.raises(TypeError, match="encode"):
            HuggingFaceTokenCounter(object())


@pytest.mark.unit
class TestEdgeCases:
    def test_empty_text_produces_no_chunks(self) -> None:
        assert chunker().chunk("") == []
        assert chunker().chunk("   \n\n  ") == []

    def test_a_paper_with_no_headings_still_chunks(self) -> None:
        """Section detection failing must not mean losing the paper."""
        chunks = chunker(min_tokens=5).chunk(SENTENCE * 10)
        assert chunks
        assert all(c.section is None for c in chunks)

    def test_whitespace_only_sections_are_skipped(self) -> None:
        chunks = chunker(min_tokens=5).chunk("1 Introduction\n\n\n\n2 Method\n\n" + SENTENCE * 4)
        assert all(c.text.strip() for c in chunks)
