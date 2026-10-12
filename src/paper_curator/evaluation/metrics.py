"""Retrieval quality metrics.

Four metrics, because each answers a different question and no single one is
sufficient:

* **Precision@K** -- of what we returned, how much was relevant? Penalises
  noise in the results the user actually sees.
* **Recall@K** -- of what was relevant, how much did we return? Penalises
  misses. The ceiling on everything downstream: a paper retrieval never finds
  cannot be re-ranked into view, cited, or used to answer.
* **MRR** -- how far down is the *first* relevant result? Captures the
  experience of someone who reads the top hit and stops.
* **nDCG@K** -- rank-weighted, so a relevant document at position 1 counts for
  more than the same document at position 9.

Optimising one alone goes wrong predictably. Returning a hundred results makes
recall look excellent and precision collapse; returning one makes the reverse.
Reporting all four together is what stops a tuning change from looking like an
improvement when it only moved the problem.

**Relevance is judged at paper level**, not chunk level. The honest reason:
producing trustworthy chunk-level labels means a human deciding, for every one
of 4,931 chunks, whether it answers each query -- and labels that are not
trustworthy produce numbers that are worse than no numbers. Paper-level
relevance is a judgement a person can actually make and defend, and it is the
right granularity anyway, because a citation points at a paper.

All functions take a ranked list of identifiers and a set of relevant ones, so
they can be tested against hand-worked examples with no retrieval involved.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _validate_k(k: int) -> None:
    if k < 1:
        msg = f"k must be at least 1, got {k}"
        raise ValueError(msg)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top K results that are relevant.

    Divided by K rather than by the number returned, deliberately. A system
    that returns three results of which all three are relevant has not achieved
    precision 1.0 at K=10 -- it left seven slots empty, and a user asking for
    ten got three.
    """
    _validate_k(k)
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant items that appear in the top K.

    The metric that matters most for a RAG system. Generation can only work
    with what retrieval supplied, so a relevant paper missing from the top K is
    gone -- no prompt, no re-ranker and no larger model recovers it.
    """
    _validate_k(k)
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result, or 0 when none is found.

    Models the user who reads the top result and stops. Moving a relevant
    document from position 2 to position 1 raises this from 0.5 to 1.0; moving
    it from 9 to 8 barely registers, which is the right shape for the
    behaviour it describes.
    """
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def dcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Discounted cumulative gain over the top K, with binary relevance.

    Each relevant document contributes ``1 / log2(rank + 1)``, so gain decays
    with depth. Binary rather than graded relevance because the dataset labels
    papers as relevant or not; graded labels would need a human to rank degrees
    of relevance, which is far harder to do consistently.
    """
    _validate_k(k)
    return sum(
        1.0 / math.log2(index + 1)
        for index, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """DCG normalised by the best achievable DCG for this query.

    Normalisation is what makes scores comparable across queries. A query with
    one relevant paper and a query with six have very different maximum DCGs,
    so raw DCG would make the second look better regardless of ranking quality.
    """
    _validate_k(k)
    if not relevant:
        return 0.0

    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(len(relevant), k) + 1))
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / ideal


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 when at least one relevant item is in the top K, else 0.0.

    Coarse, and useful precisely because of that: averaged over a dataset it is
    the share of questions for which the system found *anything* usable. A
    system can have mediocre precision and still be valuable; one that returns
    nothing relevant for a third of queries cannot.
    """
    _validate_k(k)
    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def evaluate_query(
    ranked: Sequence[str], relevant: set[str], *, k_values: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    """Compute every metric for one query, at several cut-offs.

    Several K values because the right one depends on use: K=1 describes the
    top answer, K=5 describes what fits in a context window, K=10 describes a
    results page. A change that helps at one cut-off and hurts at another is
    worth seeing rather than averaging away.
    """
    scores: dict[str, float] = {"mrr": reciprocal_rank(ranked, relevant)}

    for k in k_values:
        scores[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
        scores[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
        scores[f"hit_rate@{k}"] = hit_rate_at_k(ranked, relevant, k)

    return scores


def aggregate(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    """Average each metric across queries.

    A plain mean, with every query weighted equally. Weighting by number of
    relevant documents would let a handful of well-covered topics dominate the
    headline number and hide poor performance on the rest.
    """
    if not per_query:
        return {}

    names = sorted({name for scores in per_query for name in scores})
    return {
        name: sum(scores.get(name, 0.0) for scores in per_query) / len(per_query) for name in names
    }
