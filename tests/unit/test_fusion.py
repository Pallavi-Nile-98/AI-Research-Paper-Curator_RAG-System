"""Tests for rank fusion and result shaping.

Fusion is pure logic over ranked lists, so it is tested against fixed inputs
with no cluster, no embedding model and no network. That matters beyond speed:
these are the only tests that can assert *exact* fused scores, and an exact
assertion is what catches a subtly wrong constant that would otherwise show up
as slightly-worse retrieval nobody traces back here.
"""

from __future__ import annotations

import datetime as dt

import pytest

from paper_curator.retrieval.fusion import (
    deduplicate_chunks,
    limit_per_paper,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from paper_curator.retrieval.models import (
    RetrievalFilters,
    RetrievedChunk,
    RetrieverContribution,
    RetrieverKind,
)


def chunk(
    chunk_id: str,
    *,
    score: float = 1.0,
    rank: int = 1,
    retriever: RetrieverKind = RetrieverKind.KEYWORD,
    arxiv_id: str = "2401.00001",
    text: str = "some passage text about retrieval systems",
) -> RetrievedChunk:
    """Build a chunk with one retriever contribution."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        arxiv_id=arxiv_id,
        version=1,
        title="A Paper",
        authors=["Jane Doe"],
        abs_url="u",
        pdf_url="u",
        published_at="2024-01-01T00:00:00+00:00",
        chunk_index=0,
        contributions=[RetrieverContribution(retriever=retriever, rank=rank, score=score)],
    )


@pytest.mark.unit
class TestReciprocalRankFusion:
    def test_score_matches_the_formula(self) -> None:
        """RRF(d) = sum of 1 / (k + rank) over the retrievers that found it."""
        fused = reciprocal_rank_fusion([[chunk("a", rank=1)]], k=60)
        assert fused[0].score == pytest.approx(1 / 61)

    def test_a_chunk_found_by_both_outranks_one_found_by_either(self) -> None:
        """The core behaviour: corroboration by two signals beats one.

        The shared chunk is ranked 3rd by both retrievers, while the others are
        ranked 1st by a single retriever. Two third places still win.
        """
        keyword = [chunk("solo-k", rank=1), chunk("shared", rank=3)]
        vector = [
            chunk("solo-v", rank=1, retriever=RetrieverKind.VECTOR),
            chunk("shared", rank=3, retriever=RetrieverKind.VECTOR),
        ]

        fused = reciprocal_rank_fusion([keyword, vector], k=60)

        assert fused[0].chunk_id == "shared"
        assert fused[0].score == pytest.approx(2 / 63)

    def test_a_chunk_found_by_one_retriever_still_surfaces(self) -> None:
        """Agreement is rewarded, but disagreement is not disqualifying.

        A passage only the vector retriever finds is exactly the case hybrid
        retrieval exists for.
        """
        fused = reciprocal_rank_fusion(
            [[chunk("k-only", rank=1)], [chunk("v-only", rank=1, retriever=RetrieverKind.VECTOR)]],
            k=60,
        )
        assert {c.chunk_id for c in fused} == {"k-only", "v-only"}

    def test_raw_scores_do_not_affect_the_outcome(self) -> None:
        """The whole point: BM25 and cosine scales are never compared.

        A BM25 score of 40 and a cosine of 0.4 at the same rank contribute
        identically.
        """
        huge = reciprocal_rank_fusion([[chunk("a", score=987.0, rank=1)]], k=60)
        tiny = reciprocal_rank_fusion([[chunk("b", score=0.0001, rank=1)]], k=60)
        assert huge[0].score == pytest.approx(tiny[0].score)

    def test_provenance_survives_fusion(self) -> None:
        """A fused result must still explain where it came from."""
        fused = reciprocal_rank_fusion(
            [
                [chunk("shared", rank=2, score=30.0)],
                [chunk("shared", rank=5, score=0.8, retriever=RetrieverKind.VECTOR)],
            ],
            k=60,
        )
        found_by = fused[0].found_by
        assert RetrieverKind.KEYWORD in found_by
        assert RetrieverKind.VECTOR in found_by
        assert "keyword#2" in fused[0].explain()
        assert "vector#5" in fused[0].explain()

    def test_results_are_ordered_best_first(self) -> None:
        keyword = [chunk("a", rank=1), chunk("b", rank=2), chunk("c", rank=3)]
        fused = reciprocal_rank_fusion([keyword], k=60)
        assert [c.chunk_id for c in fused] == ["a", "b", "c"]

    def test_larger_k_flattens_the_curve(self) -> None:
        """Reduces how much the top few ranks dominate."""
        small = reciprocal_rank_fusion([[chunk("a", rank=1), chunk("b", rank=10)]], k=1)
        large = reciprocal_rank_fusion([[chunk("a", rank=1), chunk("b", rank=10)]], k=1000)

        small_gap = small[0].score / small[1].score
        large_gap = large[0].score / large[1].score
        assert small_gap > large_gap

    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_rejects_a_nonsensical_k(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            reciprocal_rank_fusion([[chunk("a")]], k=0)


@pytest.mark.unit
class TestWeightedScoreFusion:
    def test_scores_are_normalised_before_combining(self) -> None:
        """BM25's 40 and cosine's 0.9 both become 1.0 at the top of their list."""
        keyword = [chunk("a", score=40.0, rank=1), chunk("b", score=10.0, rank=2)]
        vector = [
            chunk("a", score=0.9, rank=1, retriever=RetrieverKind.VECTOR),
            chunk("b", score=0.1, rank=2, retriever=RetrieverKind.VECTOR),
        ]

        fused = weighted_score_fusion([keyword, vector], weights=[0.5, 0.5])

        by_id = {c.chunk_id: c.score for c in fused}
        assert by_id["a"] == pytest.approx(1.0)
        assert by_id["b"] == pytest.approx(0.0)

    def test_weights_shift_the_balance(self) -> None:
        keyword = [chunk("k", score=10.0, rank=1), chunk("v", score=0.0, rank=2)]
        vector = [
            chunk("v", score=10.0, rank=1, retriever=RetrieverKind.VECTOR),
            chunk("k", score=0.0, rank=2, retriever=RetrieverKind.VECTOR),
        ]

        keyword_heavy = weighted_score_fusion([keyword, vector], weights=[0.9, 0.1])
        vector_heavy = weighted_score_fusion([keyword, vector], weights=[0.1, 0.9])

        assert keyword_heavy[0].chunk_id == "k"
        assert vector_heavy[0].chunk_id == "v"

    def test_identical_scores_do_not_divide_by_zero(self) -> None:
        """Every document equally good by that retriever's reckoning."""
        same = [chunk("a", score=5.0, rank=1), chunk("b", score=5.0, rank=2)]
        fused = weighted_score_fusion([same], weights=[1.0])
        assert all(c.score == pytest.approx(1.0) for c in fused)

    def test_absence_is_not_treated_as_a_zero_score(self) -> None:
        """A chunk missing from a list simply contributes nothing for it.

        Scoring it zero would penalise a document for not being returned,
        which is different from being returned last.
        """
        keyword = [chunk("only-k", score=10.0, rank=1)]
        vector = [chunk("only-v", score=10.0, rank=1, retriever=RetrieverKind.VECTOR)]
        fused = weighted_score_fusion([keyword, vector], weights=[0.5, 0.5])
        assert all(c.score == pytest.approx(0.5) for c in fused)

    def test_rejects_mismatched_weights(self) -> None:
        with pytest.raises(ValueError, match="expected 2 weights"):
            weighted_score_fusion([[chunk("a")], [chunk("b")]], weights=[1.0])


@pytest.mark.unit
class TestDeduplication:
    def test_near_identical_passages_are_collapsed(self) -> None:
        """Chunk overlap guarantees adjacent chunks share text."""
        text = "the quick brown fox jumps over the lazy dog every single day"
        chunks = [
            chunk("a", text=text, score=2.0),
            chunk("b", text=text, score=1.0),
        ]
        assert len(deduplicate_chunks(chunks, threshold=0.9)) == 1

    def test_the_higher_scoring_duplicate_is_kept(self) -> None:
        """Input arrives ordered best-first, so the first seen is the best."""
        text = "identical passage text here for comparison purposes only"
        kept = deduplicate_chunks(
            [chunk("best", text=text, score=9.0), chunk("worse", text=text, score=1.0)],
            threshold=0.9,
        )
        assert kept[0].chunk_id == "best"

    def test_distinct_passages_are_both_kept(self) -> None:
        kept = deduplicate_chunks(
            [
                chunk("a", text="retrieval augmented generation over papers"),
                chunk("b", text="a completely different topic about compilers"),
            ],
            threshold=0.9,
        )
        assert len(kept) == 2

    def test_threshold_controls_strictness(self) -> None:
        chunks = [
            chunk("a", text="one two three four five six seven eight"),
            chunk("b", text="one two three four five six seven nine"),
        ]
        assert len(deduplicate_chunks(chunks, threshold=0.99)) == 2
        assert len(deduplicate_chunks(chunks, threshold=0.5)) == 1

    def test_empty_input(self) -> None:
        assert deduplicate_chunks([]) == []


@pytest.mark.unit
class TestPerPaperLimit:
    def test_caps_chunks_from_one_paper(self) -> None:
        """Otherwise one relevant paper answers a multi-source question alone."""
        chunks = [chunk(f"c{n}", arxiv_id="2401.00001") for n in range(8)]
        assert len(limit_per_paper(chunks, max_per_paper=3)) == 3

    def test_other_papers_are_unaffected(self) -> None:
        chunks = [
            *[chunk(f"a{n}", arxiv_id="2401.00001") for n in range(5)],
            *[chunk(f"b{n}", arxiv_id="2401.00002") for n in range(5)],
        ]
        kept = limit_per_paper(chunks, max_per_paper=2)
        assert len(kept) == 4
        assert {c.arxiv_id for c in kept} == {"2401.00001", "2401.00002"}

    def test_the_best_chunks_from_each_paper_are_kept(self) -> None:
        """Input is ordered best-first, and that order is preserved."""
        chunks = [chunk(f"c{n}", arxiv_id="2401.00001", score=10.0 - n) for n in range(5)]
        kept = limit_per_paper(chunks, max_per_paper=2)
        assert [c.chunk_id for c in kept] == ["c0", "c1"]

    def test_rejects_a_nonsensical_cap(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            limit_per_paper([chunk("a")], max_per_paper=0)


@pytest.mark.unit
class TestRetrievalFilters:
    def test_empty_filters_produce_no_clauses(self) -> None:
        filters = RetrievalFilters()
        assert filters.is_empty
        assert filters.to_opensearch_clauses() == []

    def test_categories_filter_on_the_keyword_field(self) -> None:
        clauses = RetrievalFilters(categories=["cs.CL"]).to_opensearch_clauses()
        assert clauses == [{"terms": {"categories": ["cs.CL"]}}]

    def test_authors_match_exactly_rather_than_by_token(self) -> None:
        """Filtering on a text field would match any document mentioning Jane."""
        clauses = RetrievalFilters(authors=["Jane Doe"]).to_opensearch_clauses()
        assert clauses == [{"terms": {"authors.keyword": ["Jane Doe"]}}]

    def test_a_date_range_becomes_one_range_clause(self) -> None:
        clauses = RetrievalFilters(
            published_after=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
            published_before=dt.datetime(2024, 12, 31, tzinfo=dt.UTC),
        ).to_opensearch_clauses()
        assert len(clauses) == 1
        bounds = clauses[0]["range"]["published_at"]
        assert "gte" in bounds
        assert "lte" in bounds

    def test_an_open_ended_range_is_allowed(self) -> None:
        clauses = RetrievalFilters(
            published_after=dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
        ).to_opensearch_clauses()
        assert "lte" not in clauses[0]["range"]["published_at"]

    def test_several_filters_combine(self) -> None:
        clauses = RetrievalFilters(
            arxiv_ids=["2401.00001"], categories=["cs.IR"]
        ).to_opensearch_clauses()
        assert len(clauses) == 2
