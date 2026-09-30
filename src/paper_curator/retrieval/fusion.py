"""Combining two ranked lists into one.

The problem fusion solves is that BM25 scores and cosine similarities are not
comparable. BM25 is unbounded and depends on corpus statistics -- a score of 12
means nothing without knowing the collection. Cosine similarity is bounded in
[-1, 1]. Adding them, averaging them, or picking the larger is meaningless
arithmetic that nonetheless produces a number and an ordering.

Two ways out, both implemented so the choice can be measured rather than
assumed:

**Reciprocal Rank Fusion** discards scores entirely and uses rank position::

    RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))

Nothing needs normalising, because nothing is compared across scales. A
document ranked well by both retrievers accumulates from both; one ranked well
by a single retriever still surfaces. The constant ``k`` damps how much the
very top ranks dominate -- 60 is the value from the original paper.

**Weighted score fusion** min-max normalises each list into [0, 1] and takes a
weighted sum. It keeps information RRF throws away: that a document won by a
landslide rather than a hair. The cost is that min-max normalisation is
sensitive to the window it is computed over, so the same document can normalise
differently depending on what else happened to be retrieved.

RRF is the default because it has no tuning surface to get wrong. Which is
actually better on this corpus is a Phase 2 measurement, not an assumption.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from paper_curator.core.logging import get_logger
from paper_curator.retrieval.models import RetrievedChunk

logger = get_logger(__name__)


def _merge_contributions(chunks: Sequence[RetrievedChunk]) -> RetrievedChunk:
    """Collapse the same chunk seen by several retrievers into one record.

    Keeps every contribution, so the result can still explain that it was
    ranked first by BM25 and eighth by vector search.
    """
    base = chunks[0]
    contributions = [c for chunk in chunks for c in chunk.contributions]
    return replace(base, contributions=contributions)


def _group_by_chunk_id(
    lists: Sequence[Sequence[RetrievedChunk]],
) -> dict[str, list[RetrievedChunk]]:
    """Gather the same chunk from across several result lists."""
    grouped: dict[str, list[RetrievedChunk]] = {}
    for result_list in lists:
        for chunk in result_list:
            grouped.setdefault(chunk.chunk_id, []).append(chunk)
    return grouped


def reciprocal_rank_fusion(
    lists: Sequence[Sequence[RetrievedChunk]], *, k: int = 60
) -> list[RetrievedChunk]:
    """Fuse ranked lists by reciprocal rank.

    Args:
        lists: Result lists, each ordered best-first.
        k: Damping constant. Larger values flatten the contribution curve,
            reducing how much the top few ranks dominate.

    Returns:
        One list ordered by fused score, best first.

    """
    if k < 1:
        msg = f"k must be at least 1, got {k}"
        raise ValueError(msg)

    grouped = _group_by_chunk_id(lists)
    fused: list[RetrievedChunk] = []

    for occurrences in grouped.values():
        merged = _merge_contributions(occurrences)
        # Sum over every retriever that returned this chunk. Using the rank
        # recorded by each retriever, not the position in any combined list.
        score = sum(1.0 / (k + contribution.rank) for contribution in merged.contributions)
        fused.append(replace(merged, score=score))

    fused.sort(key=lambda chunk: chunk.score, reverse=True)
    return fused


def _min_max_normalise(scores: Sequence[float]) -> list[float]:
    """Scale scores into [0, 1].

    When every score is identical the range is zero, and there is no
    information to preserve -- every document is equally good by that
    retriever's reckoning, so they all map to 1.0 rather than dividing by zero.
    """
    if not scores:
        return []
    lowest, highest = min(scores), max(scores)
    if highest == lowest:
        return [1.0] * len(scores)
    span = highest - lowest
    return [(score - lowest) / span for score in scores]


def weighted_score_fusion(
    lists: Sequence[Sequence[RetrievedChunk]],
    *,
    weights: Sequence[float],
) -> list[RetrievedChunk]:
    """Fuse ranked lists by weighted, normalised score.

    Unlike RRF this preserves score magnitude, so a decisive win counts for
    more than a narrow one. The tradeoff is that min-max normalisation depends
    on the retrieved window: the same document normalises differently according
    to what else came back alongside it.

    A chunk missing from a list contributes nothing for that retriever rather
    than a zero, so being absent is not treated as being ranked last.
    """
    if len(weights) != len(lists):
        msg = f"expected {len(lists)} weights, got {len(weights)}"
        raise ValueError(msg)

    normalised_lookup: list[dict[str, float]] = []
    for result_list in lists:
        scores = _min_max_normalise([chunk.score for chunk in result_list])
        normalised_lookup.append(
            {chunk.chunk_id: score for chunk, score in zip(result_list, scores, strict=True)}
        )

    grouped = _group_by_chunk_id(lists)
    fused: list[RetrievedChunk] = []

    for chunk_id, occurrences in grouped.items():
        merged = _merge_contributions(occurrences)
        score = sum(
            weight * lookup[chunk_id]
            for weight, lookup in zip(weights, normalised_lookup, strict=True)
            if chunk_id in lookup
        )
        fused.append(replace(merged, score=score))

    fused.sort(key=lambda chunk: chunk.score, reverse=True)
    return fused


def deduplicate_chunks(
    chunks: Sequence[RetrievedChunk], *, threshold: float = 0.9
) -> list[RetrievedChunk]:
    """Drop near-duplicate passages, keeping the higher-scoring one.

    Chunk overlap guarantees that adjacent chunks share text by construction,
    so a question answered at a chunk boundary retrieves two passages saying
    much the same thing. Both would occupy context budget that a second source
    could have used.

    Similarity is Jaccard overlap over word sets -- deliberately cheap. This
    runs on every query over a few dozen candidates, and an embedding-based
    comparison would cost more than the retrieval it is cleaning up.
    """
    if not chunks:
        return []

    kept: list[RetrievedChunk] = []
    kept_tokens: list[set[str]] = []

    for chunk in chunks:
        tokens = set(chunk.text.lower().split())
        if not tokens:
            continue

        duplicate = False
        for existing in kept_tokens:
            union = tokens | existing
            if union and len(tokens & existing) / len(union) >= threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(chunk)
            kept_tokens.append(tokens)

    if len(kept) < len(chunks):
        logger.debug("duplicates_removed", before=len(chunks), after=len(kept))
    return kept


def limit_per_paper(
    chunks: Sequence[RetrievedChunk], *, max_per_paper: int
) -> list[RetrievedChunk]:
    """Cap how many chunks any single paper may contribute.

    Without this, one highly relevant paper fills every slot, and a question
    that needs two sources is answered from one source eight times. The cap
    trades a little depth for the breadth that multi-source questions require.

    Order is preserved, so the chunks kept from each paper are its best.
    """
    if max_per_paper < 1:
        msg = f"max_per_paper must be at least 1, got {max_per_paper}"
        raise ValueError(msg)

    seen: dict[str, int] = {}
    kept: list[RetrievedChunk] = []

    for chunk in chunks:
        count = seen.get(chunk.arxiv_id, 0)
        if count >= max_per_paper:
            continue
        seen[chunk.arxiv_id] = count + 1
        kept.append(chunk)

    return kept
