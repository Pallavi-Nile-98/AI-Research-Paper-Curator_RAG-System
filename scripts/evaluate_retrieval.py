"""Run the retrieval experiments and record the results.

    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --configs bm25 hybrid
    python scripts/evaluate_retrieval.py --report evaluation/reports/run.md

Compares four configurations over the labelled dataset:

    bm25             lexical retrieval alone
    dense            embedding similarity alone
    hybrid           both, fused by reciprocal rank
    hybrid_rerank    hybrid, then re-scored by a cross-encoder

Requires PostgreSQL and OpenSearch running with an ingested corpus, and the
optional ``[ml]`` extra for the embedding and re-ranking models.

Nothing here decides what the numbers mean. It records what happened, including
when the result is unflattering, and writes the dataset's review status into
every artefact so a provisional number cannot later be mistaken for a verified
one.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from paper_curator.core.config import RerankerSettings, Settings, get_settings
from paper_curator.core.logging import configure_logging
from paper_curator.embedding import get_embedding_provider
from paper_curator.evaluation.dataset import EvaluationDataset, load_dataset
from paper_curator.evaluation.experiment import (
    EVAL_CHUNK_LIMIT,
    ExperimentResult,
    new_result,
    save_results,
    score_example,
)
from paper_curator.retrieval import RerankingService, build_retriever
from paper_curator.search import create_client

DEFAULT_DATASET = Path("evaluation/datasets/retrieval_eval_v0.1-draft.json")
DEFAULT_RESULTS = Path("evaluation/experiments/retrieval_experiments.json")
DEFAULT_REPORT = Path("evaluation/reports/retrieval_evaluation.md")

CONFIGURATIONS = ("bm25", "dense", "hybrid", "hybrid_rerank")
LINE = "=" * 78

HEADLINE = ("recall@5", "recall@10", "ndcg@10", "mrr", "hit_rate@5")


async def run_configuration(
    name: str, dataset: EvaluationDataset, settings: Settings
) -> ExperimentResult:
    """Evaluate one configuration over every example in the dataset."""
    mode = {"bm25": "keyword", "dense": "vector"}.get(name, "hybrid")
    reranking = name == "hybrid_rerank"

    run_settings = settings.model_copy(
        update={
            "reranker": RerankerSettings(**{**settings.reranker.model_dump(), "enabled": reranking})
        }
    )

    client = create_client(run_settings)
    embedder = get_embedding_provider(run_settings)
    retriever = build_retriever(client, embedder, run_settings, mode=mode)
    reranker = RerankingService(settings=run_settings) if reranking else None

    # Before anything is timed. Loading takes tens of seconds; embedding takes
    # milliseconds. An unwarmed first query would dominate the distribution.
    print(f"  warming up models for {name} ...", end="", flush=True)
    warm_started = time.perf_counter()
    embedder.warmup()
    if reranker is not None:
        reranker.warmup()
    print(f" {int((time.perf_counter() - warm_started) * 1000)} ms (excluded from results)")

    result = new_result(name, run_settings, dataset)

    for index, example in enumerate(dataset.examples, start=1):
        retrieved = await retriever.retrieve(example.query, limit=EVAL_CHUNK_LIMIT)
        chunks = retrieved.chunks
        rerank_ms: int | None = None

        if reranker is not None:
            outcome = await reranker.rerank(example.query, list(chunks), top_k=EVAL_CHUNK_LIMIT)
            chunks = outcome.chunks
            rerank_ms = outcome.rerank_ms

        result.per_query.append(
            score_example(
                example,
                chunks,
                retrieval_ms=retrieved.retrieval_ms,
                rerank_ms=rerank_ms,
            )
        )
        print(f"\r  {name}: {index}/{len(dataset)} queries", end="", flush=True)

    print()
    await client.close()
    return result


def print_comparison(results: list[ExperimentResult]) -> None:
    """Print the headline comparison table."""
    print(f"\n{LINE}")
    print("OVERALL")
    print(LINE)
    header = f"{'configuration':<16}" + "".join(f"{metric:>12}" for metric in HEADLINE)
    print(header)
    print("-" * len(header))
    for result in results:
        overall = result.overall
        row = f"{result.configuration:<16}" + "".join(
            f"{overall.get(metric, 0.0):>12.3f}" for metric in HEADLINE
        )
        print(row)

    print(f"\n{LINE}")
    print("LATENCY (milliseconds)")
    print(LINE)
    print(f"{'configuration':<16}{'median':>10}{'p90':>10}{'max':>10}{'rerank':>10}")
    print("-" * 56)
    for result in results:
        stats = result.latency_stats()
        rerank = stats.get("rerank_ms_median")
        print(
            f"{result.configuration:<16}"
            f"{stats.get('retrieval_ms_median', 0):>10.0f}"
            f"{stats.get('retrieval_ms_p90', 0):>10.0f}"
            f"{stats.get('retrieval_ms_max', 0):>10.0f}"
            f"{(f'{rerank:.0f}' if rerank is not None else '-'):>10}"
        )


def print_by_category(results: list[ExperimentResult], metric: str = "recall@10") -> None:
    """Print per-category results, where the real differences live."""
    categories = sorted({c for result in results for c in result.by_category()})

    print(f"\n{LINE}")
    print(f"{metric.upper()} BY QUERY CATEGORY")
    print(LINE)
    header = f"{'category':<22}" + "".join(f"{r.configuration:>15}" for r in results)
    print(header)
    print("-" * len(header))

    for category in categories:
        row = f"{category:<22}"
        for result in results:
            scores = result.by_category().get(category, {})
            row += f"{scores.get(metric, 0.0):>15.3f}"
        print(row)


def write_report(results: list[ExperimentResult], dataset: EvaluationDataset, path: Path) -> None:
    """Write a Markdown report separating measurements from interpretation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    best = max(results, key=lambda r: r.overall.get("ndcg@10", 0.0))

    lines: list[str] = [
        "# Retrieval evaluation",
        "",
        f"Generated: {results[0].timestamp}",
        "",
        "## Dataset status",
        "",
        f"- Version: `{dataset.version}`",
        f"- {dataset.review_summary()}",
        f"- Corpus: {dataset.corpus_description}",
        "",
    ]

    if not dataset.fully_reviewed:
        lines += [
            "> **These results are provisional.** The relevance labels have not been",
            "> reviewed by a human. Every number below is measured correctly against",
            "> labels that may themselves be wrong, so no conclusion here should be",
            "> quoted as an established result.",
            "",
        ]

    lines += [
        "## Overall",
        "",
        "| configuration | " + " | ".join(HEADLINE) + " |",
        "|" + "---|" * (len(HEADLINE) + 1),
    ]
    for result in results:
        overall = result.overall
        cells = " | ".join(f"{overall.get(m, 0.0):.3f}" for m in HEADLINE)
        lines.append(f"| {result.configuration} | {cells} |")

    lines += [
        "",
        "## Latency",
        "",
        "| configuration | median ms | p90 ms | max ms | rerank median ms |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        stats = result.latency_stats()
        rerank = stats.get("rerank_ms_median")
        lines.append(
            f"| {result.configuration} | {stats.get('retrieval_ms_median', 0):.0f} | "
            f"{stats.get('retrieval_ms_p90', 0):.0f} | {stats.get('retrieval_ms_max', 0):.0f} | "
            f"{f'{rerank:.0f}' if rerank is not None else '-'} |"
        )

    lines += [
        "",
        "## Recall@10 by query category",
        "",
        "| category | " + " | ".join(r.configuration for r in results) + " |",
        "|" + "---|" * (len(results) + 1),
    ]
    categories = sorted({c for result in results for c in result.by_category()})
    for category in categories:
        cells = " | ".join(
            f"{r.by_category().get(category, {}).get('recall@10', 0.0):.3f}" for r in results
        )
        lines.append(f"| {category} | {cells} |")

    lines += [
        "",
        "## Complete misses",
        "",
        "Queries where a configuration retrieved no relevant paper at all.",
        "",
    ]
    for result in results:
        misses = result.failures(limit=10)
        lines.append(f"**{result.configuration}** — {len(misses)} shown")
        if not misses:
            lines.append("- none")
        for miss in misses:
            lines.append(f"- `{miss.example_id}` ({miss.category}): {miss.query}")
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "Measured facts are above. The following is interpretation and is labelled",
        "as such.",
        "",
        f"- Highest nDCG@10: **{best.configuration}** at {best.overall.get('ndcg@10', 0.0):.3f}.",
        "- The corpus is small and drawn from a narrow slice of cs.IR, so these",
        "  figures describe this corpus and do not generalise.",
        "- Labels are paper-level, taken from abstracts. A paper answering a question",
        "  only in its body would be scored as a miss.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "docker-compose up -d",
        "python scripts/evaluate_retrieval.py",
        "```",
        "",
        "The full configuration of every run, including model names and fusion",
        f"parameters, is recorded in `{DEFAULT_RESULTS}`.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {path}")


async def main_async(args: argparse.Namespace) -> int:
    """Run the requested configurations and record the results."""
    settings = get_settings()
    configure_logging(level=args.log_level, log_format="console")

    dataset = load_dataset(args.dataset)
    print(LINE)
    print(f"Dataset: {dataset.version} — {dataset.review_summary()}")
    print(LINE)

    if not dataset.fully_reviewed:
        print("\n  NOTE: labels are unreviewed. Results will be recorded as provisional.\n")

    results: list[ExperimentResult] = []
    for name in args.configs:
        print(f"\nRunning {name} ...")
        results.append(await run_configuration(name, dataset, settings))

    print_comparison(results)
    print_by_category(results, metric="recall@10")

    save_results(results, args.results)
    print(f"\nRaw results written to {args.results}")
    write_report(results, dataset, args.report)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run retrieval experiments.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--configs", nargs="+", default=list(CONFIGURATIONS), choices=CONFIGURATIONS
    )
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    try:
        return asyncio.run(main_async(parse_args()))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
