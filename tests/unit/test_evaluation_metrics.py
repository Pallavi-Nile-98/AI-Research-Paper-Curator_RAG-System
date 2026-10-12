"""Tests for the retrieval metrics.

These functions produce every number in the evaluation report and in ADR-0008.
If ``ndcg_at_k`` is subtly wrong, the conclusion that dense retrieval beat
hybrid is wrong too, and nothing downstream would reveal it -- a broken metric
still returns a plausible float, still ranks configurations in *some* order,
and still fills in a results table that looks exactly like a correct one.

So the expected values here are worked out by hand from the definitions rather
than captured from a run. A test that records what the code currently does
cannot detect that the code is wrong; it only detects that the code changed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import pytest

from paper_curator.evaluation.metrics import (
    aggregate,
    dcg_at_k,
    evaluate_query,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Shorthand used throughout: single-letter document ids keep the hand
# calculations legible.
RANKED = ["a", "b", "c", "d", "e"]

type MetricAtK = Callable[[Sequence[str], set[str], int], float]
"""Shape shared by every cut-off metric, for the parametrised K validation."""


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_none_relevant(self) -> None:
        assert precision_at_k(RANKED, {"z"}, k=5) == 0.0

    def test_divides_by_k_not_by_results_returned(self) -> None:
        """Three results, all relevant, K=10 is precision 0.3 -- not 1.0.

        The deliberate choice documented in the module. A user who asked for
        ten results and got three has not been served perfectly, and dividing
        by the number returned would score that as flawless.
        """
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=10) == pytest.approx(0.3)

    def test_counts_only_within_the_cutoff(self) -> None:
        # "e" is relevant but sits at rank 5, outside K=3.
        assert precision_at_k(RANKED, {"e"}, k=3) == 0.0

    def test_empty_relevant_set_is_zero(self) -> None:
        assert precision_at_k(RANKED, set(), k=5) == 0.0


class TestRecallAtK:
    def test_finds_all(self) -> None:
        assert recall_at_k(RANKED, {"a", "b"}, k=5) == 1.0

    def test_finds_half(self) -> None:
        # "a" is within K=3; "e" is not.
        assert recall_at_k(RANKED, {"a", "e"}, k=3) == pytest.approx(0.5)

    def test_divides_by_relevant_count_not_by_k(self) -> None:
        """One relevant document found at K=10 is recall 1.0, not 0.1."""
        assert recall_at_k(RANKED, {"a"}, k=10) == 1.0

    def test_empty_relevant_set_is_zero(self) -> None:
        assert recall_at_k(RANKED, set(), k=5) == 0.0


class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert reciprocal_rank(RANKED, {"a"}) == 1.0

    def test_second_position(self) -> None:
        assert reciprocal_rank(RANKED, {"b"}) == pytest.approx(0.5)

    def test_uses_the_first_relevant_only(self) -> None:
        """Later relevant documents do not improve the score."""
        assert reciprocal_rank(RANKED, {"b", "c", "d"}) == pytest.approx(0.5)

    def test_not_found_is_zero(self) -> None:
        assert reciprocal_rank(RANKED, {"z"}) == 0.0


class TestDcgAtK:
    def test_single_hit_at_rank_one(self) -> None:
        # 1 / log2(1 + 1) = 1 / 1 = 1.0
        assert dcg_at_k(RANKED, {"a"}, k=5) == pytest.approx(1.0)

    def test_single_hit_at_rank_two(self) -> None:
        # 1 / log2(2 + 1) = 1 / 1.58496 = 0.63093
        assert dcg_at_k(RANKED, {"b"}, k=5) == pytest.approx(1.0 / math.log2(3))

    def test_two_hits_sum(self) -> None:
        # ranks 1 and 3: 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        assert dcg_at_k(RANKED, {"a", "c"}, k=5) == pytest.approx(1.5)

    def test_discount_rewards_earlier_positions(self) -> None:
        """The property nDCG exists for: same document, earlier, scores more."""
        assert dcg_at_k(["a", "z"], {"a"}, k=2) > dcg_at_k(["z", "a"], {"a"}, k=2)


class TestNdcgAtK:
    def test_perfect_ranking_is_one(self) -> None:
        assert ndcg_at_k(["a", "b", "z"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_hand_worked_imperfect_ranking(self) -> None:
        """Relevant documents at ranks 1 and 3, ideal is ranks 1 and 2.

        DCG   = 1/log2(2) + 1/log2(4) = 1.0 + 0.5       = 1.5
        ideal = 1/log2(2) + 1/log2(3) = 1.0 + 0.63093   = 1.63093
        nDCG  = 1.5 / 1.63093                           = 0.91972
        """
        expected = 1.5 / (1.0 + 1.0 / math.log2(3))
        assert ndcg_at_k(RANKED, {"a", "c"}, k=5) == pytest.approx(expected)
        assert ndcg_at_k(RANKED, {"a", "c"}, k=5) == pytest.approx(0.91972, abs=1e-5)

    def test_normalisation_makes_queries_comparable(self) -> None:
        """A query with one relevant paper and one with three both reach 1.0.

        Without normalisation the second would have a far larger raw DCG and
        would dominate any average, regardless of ranking quality.
        """
        one = ndcg_at_k(["a", "z", "y"], {"a"}, k=3)
        three = ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3)
        assert one == pytest.approx(1.0)
        assert three == pytest.approx(1.0)

    def test_ideal_is_capped_at_k(self) -> None:
        """Six relevant documents evaluated at K=2 can still reach nDCG 1.0.

        The ideal ranking is the best achievable *within K*, so a query whose
        relevant set is larger than the cutoff is not penalised for the
        documents that could not fit.
        """
        relevant = {"a", "b", "c", "d", "e", "f"}
        assert ndcg_at_k(["a", "b"], relevant, k=2) == pytest.approx(1.0)

    def test_nothing_relevant_is_zero(self) -> None:
        assert ndcg_at_k(RANKED, {"z"}, k=5) == 0.0

    def test_empty_relevant_set_is_zero(self) -> None:
        assert ndcg_at_k(RANKED, set(), k=5) == 0.0

    def test_bounded_between_zero_and_one(self) -> None:
        for relevant in ({"a"}, {"e"}, {"a", "e"}, {"b", "c", "d"}):
            score = ndcg_at_k(RANKED, relevant, k=5)
            assert 0.0 <= score <= 1.0


class TestHitRateAtK:
    def test_one_hit_is_one(self) -> None:
        assert hit_rate_at_k(RANKED, {"e"}, k=5) == 1.0

    def test_insensitive_to_how_many_hits(self) -> None:
        """Deliberately coarse: one relevant result scores the same as five."""
        assert hit_rate_at_k(RANKED, {"a"}, k=5) == hit_rate_at_k(RANKED, set(RANKED), k=5)

    def test_outside_cutoff_is_zero(self) -> None:
        assert hit_rate_at_k(RANKED, {"e"}, k=3) == 0.0


class TestKValidation:
    @pytest.mark.parametrize(
        "func", [precision_at_k, recall_at_k, dcg_at_k, ndcg_at_k, hit_rate_at_k]
    )
    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_non_positive_k(self, func: MetricAtK, k: int) -> None:
        """K=0 would silently return 0.0 everywhere and look like total failure."""
        with pytest.raises(ValueError, match="k must be at least 1"):
            func(RANKED, {"a"}, k)


class TestEvaluateQuery:
    def test_reports_every_metric_at_every_cutoff(self) -> None:
        scores = evaluate_query(RANKED, {"a"}, k_values=(1, 3))
        assert set(scores) == {
            "mrr",
            "precision@1",
            "recall@1",
            "ndcg@1",
            "hit_rate@1",
            "precision@3",
            "recall@3",
            "ndcg@3",
            "hit_rate@3",
        }

    def test_values_match_the_individual_functions(self) -> None:
        scores = evaluate_query(RANKED, {"a", "c"}, k_values=(5,))
        assert scores["recall@5"] == recall_at_k(RANKED, {"a", "c"}, 5)
        assert scores["ndcg@5"] == ndcg_at_k(RANKED, {"a", "c"}, 5)
        assert scores["mrr"] == reciprocal_rank(RANKED, {"a", "c"})

    def test_complete_miss_scores_zero_throughout(self) -> None:
        scores = evaluate_query(RANKED, {"z"}, k_values=(1, 5))
        assert all(value == 0.0 for value in scores.values())


class TestAggregate:
    def test_plain_mean(self) -> None:
        result = aggregate([{"recall@5": 1.0}, {"recall@5": 0.0}])
        assert result["recall@5"] == pytest.approx(0.5)

    def test_every_query_weighted_equally(self) -> None:
        """A query with six relevant papers counts once, same as one with one.

        Weighting by relevant-document count would let a few well-covered
        topics set the headline number.
        """
        result = aggregate([{"recall@10": 1.0}, {"recall@10": 1.0}, {"recall@10": 0.0}])
        assert result["recall@10"] == pytest.approx(2 / 3)

    def test_missing_metric_counts_as_zero_not_skipped(self) -> None:
        """A query missing a metric must not be dropped from its denominator.

        Averaging only over the queries that have the key would quietly report
        a mean over a smaller set than the dataset, inflating the result.
        """
        result = aggregate([{"recall@5": 1.0}, {"mrr": 1.0}])
        assert result["recall@5"] == pytest.approx(0.5)

    def test_empty_input(self) -> None:
        assert aggregate([]) == {}
