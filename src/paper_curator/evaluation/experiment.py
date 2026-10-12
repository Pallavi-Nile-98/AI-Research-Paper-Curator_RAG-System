"""Running reproducible retrieval experiments.

An experiment is a configuration plus a dataset plus the resulting numbers,
recorded together. Keeping them together is the point: a metric without the
configuration that produced it cannot be reproduced, compared, or defended --
and a number nobody can reproduce should not appear on a resume.

**Chunk rankings are converted to paper rankings** before scoring, because
relevance is labelled at paper level. A paper's rank is the position of its
best-ranked chunk. Two consequences worth stating plainly:

* The number of distinct papers a query can reach is bounded by the chunk pool
  and the per-paper cap. With 30 chunks and at most 3 per paper, a query
  reaches at most 10 papers -- which is why metrics are reported to K=10 and no
  further.
* A system that returns ten excellent chunks from one paper scores poorly on
  recall for a multi-source question, and should: that question needs several
  papers, and one paper ten times does not answer it.

**The model is warmed up before anything is timed.** Loading the embedding
model takes roughly forty seconds while embedding a query takes milliseconds,
so an unwarmed first query would put a forty-second outlier in the latency
distribution and make the reported figure meaningless.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from paper_curator.core.config import Settings
from paper_curator.core.logging import get_logger
from paper_curator.evaluation.dataset import EvaluationDataset, EvaluationExample
from paper_curator.evaluation.metrics import aggregate, evaluate_query
from paper_curator.retrieval.models import RetrievedChunk

logger = get_logger(__name__)

# Chunks retrieved per query before mapping to papers. Chosen with the
# per-paper cap so that ten distinct papers are reachable, which is what makes
# metrics at K=10 meaningful rather than structurally capped.
EVAL_CHUNK_LIMIT = 30

K_VALUES = (1, 3, 5, 10)


def chunks_to_paper_ranking(chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Reduce a ranked chunk list to a ranked list of distinct papers.

    A paper takes the position of its best chunk, and later chunks from the
    same paper do not push other papers down -- otherwise one paper
    contributing three chunks would occupy three of the ten slots and
    mechanically depress recall for every multi-source question.
    """
    seen: set[str] = set()
    ranking: list[str] = []
    for chunk in chunks:
        if chunk.arxiv_id not in seen:
            seen.add(chunk.arxiv_id)
            ranking.append(chunk.arxiv_id)
    return ranking


@dataclass(frozen=True, slots=True)
class QueryResult:
    """What one configuration did on one query."""

    example_id: str
    query: str
    category: str
    relevant: list[str]
    retrieved_papers: list[str]
    metrics: dict[str, float]
    retrieval_ms: int
    rerank_ms: int | None = None

    @property
    def found_any(self) -> bool:
        return bool(set(self.retrieved_papers) & set(self.relevant))


@dataclass
class ExperimentResult:
    """One configuration evaluated over the whole dataset."""

    configuration: str
    config_snapshot: dict[str, Any]
    dataset_version: str
    dataset_review_status: str
    timestamp: str
    per_query: list[QueryResult] = field(default_factory=list)

    @property
    def overall(self) -> dict[str, float]:
        return aggregate([result.metrics for result in self.per_query])

    def by_category(self) -> dict[str, dict[str, float]]:
        """Metrics grouped by query category.

        Where the comparison actually lives. BM25 and dense retrieval are
        expected to diverge on specific categories, and an overall average can
        leave two very different systems looking equivalent.
        """
        grouped: dict[str, list[dict[str, float]]] = {}
        for result in self.per_query:
            grouped.setdefault(result.category, []).append(result.metrics)
        return {category: aggregate(scores) for category, scores in sorted(grouped.items())}

    def latency_stats(self) -> dict[str, float]:
        """Latency distribution, reported rather than averaged.

        A mean hides the tail, and the tail is what a user notices.
        """
        retrieval = sorted(result.retrieval_ms for result in self.per_query)
        if not retrieval:
            return {}

        stats = {
            "retrieval_ms_median": float(statistics.median(retrieval)),
            "retrieval_ms_mean": float(statistics.fmean(retrieval)),
            "retrieval_ms_min": float(retrieval[0]),
            "retrieval_ms_max": float(retrieval[-1]),
            "retrieval_ms_p90": float(
                retrieval[min(len(retrieval) - 1, int(len(retrieval) * 0.9))]
            ),
        }

        reranks = [r.rerank_ms for r in self.per_query if r.rerank_ms is not None]
        if reranks:
            ordered = sorted(reranks)
            stats["rerank_ms_median"] = float(statistics.median(ordered))
            stats["rerank_ms_max"] = float(ordered[-1])
            stats["total_ms_median"] = stats["retrieval_ms_median"] + stats["rerank_ms_median"]

        return stats

    def failures(self, limit: int = 5) -> list[QueryResult]:
        """Return queries where nothing relevant was retrieved at all.

        The most informative rows in the whole experiment: a configuration's
        complete misses say more about its weaknesses than its average score.
        """
        return [result for result in self.per_query if not result.found_any][:limit]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full result, including every per-query outcome."""
        return {
            "configuration": self.configuration,
            "config_snapshot": self.config_snapshot,
            "dataset_version": self.dataset_version,
            "dataset_review_status": self.dataset_review_status,
            "timestamp": self.timestamp,
            "overall": self.overall,
            "by_category": self.by_category(),
            "latency": self.latency_stats(),
            "per_query": [asdict(result) for result in self.per_query],
        }


def build_config_snapshot(settings: Settings, configuration: str) -> dict[str, Any]:
    """Capture everything needed to reproduce a run.

    Deliberately verbose. Months later, "hybrid scored 0.72" is not a
    reproducible claim; "hybrid with RRF k=60, bge-small-en-v1.5, a 50-chunk
    pool and at most 3 chunks per paper scored 0.72" is.
    """
    return {
        "configuration": configuration,
        "retrieval": {
            "candidate_pool_size": settings.retrieval.candidate_pool_size,
            "fusion_method": settings.retrieval.fusion_method,
            "rrf_k": settings.retrieval.rrf_k,
            "keyword_weight": settings.retrieval.keyword_weight,
            "vector_weight": settings.retrieval.vector_weight,
            "bm25_text_boost": settings.retrieval.bm25_text_boost,
            "bm25_exact_boost": settings.retrieval.bm25_exact_boost,
            "bm25_title_boost": settings.retrieval.bm25_title_boost,
            "max_chunks_per_paper": settings.retrieval.max_chunks_per_paper,
            "deduplicate": settings.retrieval.deduplicate,
            "eval_chunk_limit": EVAL_CHUNK_LIMIT,
        },
        "embedding": {
            "model": settings.embedding.model_name,
            "dimensions": settings.embedding.dimensions,
            "normalize": settings.embedding.normalize,
            "device": settings.embedding.device,
        },
        "reranker": {
            "enabled": settings.reranker.enabled,
            "model": settings.reranker.model_name,
            "top_k": settings.reranker.top_k,
            "device": settings.reranker.device,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
    }


def score_example(
    example: EvaluationExample,
    chunks: Sequence[RetrievedChunk],
    *,
    retrieval_ms: int,
    rerank_ms: int | None = None,
) -> QueryResult:
    """Score one configuration's output for one query."""
    ranking = chunks_to_paper_ranking(chunks)
    return QueryResult(
        example_id=example.id,
        query=example.query,
        category=example.category.value,
        relevant=list(example.relevant_arxiv_ids),
        retrieved_papers=ranking,
        metrics=evaluate_query(ranking, example.relevant_set, k_values=K_VALUES),
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
    )


def new_result(
    configuration: str, settings: Settings, dataset: EvaluationDataset
) -> ExperimentResult:
    """Start an empty result, stamped with its configuration and dataset."""
    return ExperimentResult(
        configuration=configuration,
        config_snapshot=build_config_snapshot(settings, configuration),
        dataset_version=dataset.version,
        # Carried into every artefact, so a provisional number cannot be
        # separated from the fact that its labels were never reviewed.
        dataset_review_status=dataset.review_summary(),
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
    )


def save_results(results: Sequence[ExperimentResult], path: Path) -> None:
    """Write experiment results to version-controlled JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": dt.datetime.now(dt.UTC).isoformat(),
        "experiments": [result.to_dict() for result in results],
    }
    # Trailing newline so regenerating results does not leave a diff for the
    # end-of-file fixer to make on every commit.
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("experiment_results_saved", path=str(path), experiments=len(results))


def timed() -> float:
    """Monotonic clock reading, in seconds."""
    return time.perf_counter()
