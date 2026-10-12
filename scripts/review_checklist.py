"""Generate a human-reviewable checklist for the evaluation dataset.

    python scripts/review_checklist.py
    python scripts/review_checklist.py --out evaluation/datasets/review-round2.md

Every label in the dataset was drafted by reading an abstract, which is a
weaker standard than reading the paper. This script turns those drafts into
something a person can actually sit down and check: each question, the papers
it marks relevant with their titles and abstracts, the reasoning behind the
label, and -- the part that matters most -- **the papers retrieval ranked
highly that are not labelled relevant**.

That last section exists because of an asymmetry in how a hand-drafted dataset
fails. A wrong positive label is visible: the paper is listed, and reading its
abstract shows it does not answer the question. A **missing** positive label is
invisible: a paper that genuinely answers the question but was never labelled
is silently counted as a false positive, which understates precision for every
configuration and can reverse a comparison. Those papers are only findable by
looking at what the system retrieved and was marked wrong for, so they are
printed here with the rank each configuration gave them.

The checklist is generated rather than written by hand so it can be
regenerated after corrections are applied, and so it cannot drift away from the
dataset it describes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import select

from paper_curator.db.models.paper import Chunk, Paper
from paper_curator.db.session import session_scope
from paper_curator.evaluation.dataset import EvaluationDataset, load_dataset

DEFAULT_DATASET = Path("evaluation/datasets/retrieval_eval_v0.1-draft.json")
DEFAULT_RESULTS = Path("evaluation/experiments/retrieval_experiments.json")
DEFAULT_OUT = Path("evaluation/datasets/LABEL_REVIEW.md")

# How many unlabelled-but-retrieved papers to surface per question. Enough to
# catch a missed label, few enough that the reviewer does not abandon the task.
CANDIDATE_LIMIT = 5

ABSTRACT_CHARS = 420


async def load_papers(arxiv_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Fetch title, abstract and chunk count for each paper."""
    async with session_scope() as session:
        rows = (
            await session.execute(select(Paper).where(Paper.arxiv_id.in_(sorted(arxiv_ids))))
        ).scalars()
        papers = {
            paper.arxiv_id: {
                "title": paper.title,
                "abstract": paper.abstract,
                "category": paper.primary_category,
                "id": paper.id,
            }
            for paper in rows
        }

        for record in papers.values():
            count = await session.scalar(
                select(Chunk.id).where(Chunk.paper_id == record["id"]).limit(1)
            )
            record["has_chunks"] = count is not None

    return papers


def load_retrieved(path: Path) -> dict[str, dict[str, list[str]]]:
    """Map example id -> configuration -> ranked paper list, if results exist.

    Missing results are not an error. The checklist is still usable without
    them; it just cannot suggest candidate labels.
    """
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    retrieved: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for experiment in payload["experiments"]:
        name = experiment["configuration"]
        for query in experiment["per_query"]:
            retrieved[query["example_id"]][name] = query["retrieved_papers"]
    return dict(retrieved)


def candidate_labels(
    per_config: dict[str, list[str]], relevant: set[str], limit: int
) -> list[tuple[str, dict[str, int]]]:
    """Papers retrieved near the top that no label marks relevant.

    Ordered by best rank achieved by any configuration, because a paper every
    system puts first and the dataset calls irrelevant is the single most
    likely place for the dataset to be wrong.
    """
    best: dict[str, dict[str, int]] = defaultdict(dict)
    for config, ranking in per_config.items():
        for position, arxiv_id in enumerate(ranking[:limit], start=1):
            if arxiv_id not in relevant:
                best[arxiv_id][config] = position

    ordered = sorted(best.items(), key=lambda item: (min(item[1].values()), item[0]))
    return ordered[:limit]


def wrap(text: str, width: int = 88, indent: str = "") -> str:
    """Wrap prose so the generated file stays readable in a diff."""
    return (
        "\n".join(
            textwrap.wrap(
                " ".join(text.split()), width=width, initial_indent=indent, subsequent_indent=indent
            )
        )
        or indent
    )


def paper_block(arxiv_id: str, papers: dict[str, dict[str, Any]], *, marker: str) -> list[str]:
    """Render one paper: title, link, and enough abstract to judge it."""
    record = papers.get(arxiv_id)
    if record is None:
        return [f"{marker} `{arxiv_id}` — **NOT IN CORPUS** (label references a missing paper)", ""]

    abstract = " ".join(record["abstract"].split())
    if len(abstract) > ABSTRACT_CHARS:
        abstract = abstract[:ABSTRACT_CHARS].rsplit(" ", 1)[0] + " ..."

    warning = "" if record["has_chunks"] else "  **NO CHUNKS INDEXED — unreachable**"
    return [
        f"{marker} **{record['title']}**{warning}",
        f"  `{arxiv_id}` · {record['category']} · [abstract](https://arxiv.org/abs/{arxiv_id})",
        "",
        wrap(abstract, width=86, indent="  > "),
        "",
    ]


def build_checklist(
    dataset: EvaluationDataset,
    papers: dict[str, dict[str, Any]],
    retrieved: dict[str, dict[str, list[str]]],
) -> str:
    """Render the whole checklist."""
    lines: list[str] = [
        "# Label review checklist",
        "",
        f"Dataset `{dataset.version}` — {dataset.review_summary()}",
        "",
        f"Corpus: {dataset.corpus_description}",
        "",
        "## How to use this",
        "",
        "Work through the questions in any order. For each one there are three things",
        "to decide, and the third is the one that is easy to skip and most worth doing.",
        "",
        "1. **Is the question reasonable?** Would someone actually ask it? A question",
        "   nobody would ask measures nothing useful, however well the system answers it.",
        "2. **Are the labelled papers actually relevant?** Read the abstract shown. If a",
        "   paper does not answer the question, strike it.",
        "3. **Is anything missing?** Under *Retrieved but not labelled* are the papers",
        "   the system ranked highly and was marked wrong for. If one of them does answer",
        "   the question, the label is missing, not the retrieval. This is the failure",
        "   mode a drafted dataset is most prone to, and the only way to catch it is to",
        "   look.",
        "",
        "Annotate however is convenient — edit this file directly, or write a list. The",
        "fields under each question are a suggestion:",
        "",
        "```",
        "VERDICT:   ok | fix | drop",
        "REMOVE:    arxiv ids that should not be labelled relevant",
        "ADD:       arxiv ids that should be",
        "REWORD:    a better version of the question, if it needs one",
        "```",
        "",
        "A `drop` verdict is a legitimate outcome. A question that turns out to be",
        "ambiguous or unanswerable is better removed than left in to add noise.",
        "",
        "---",
        "",
        "## Why this review matters for the headline result",
        "",
        "The measured result is that dense retrieval beat hybrid on this dataset. The",
        "known threat to that result is that **the questions were drafted from paper",
        "abstracts by the same process that built the system**, and paraphrase is exactly",
        "what dense retrieval is good at. If the questions are systematically closer in",
        "wording to the abstracts than a real user's question would be, the dataset",
        "favours dense retrieval by construction and the comparison is not meaningful.",
        "",
        "Watch for this while reading. A question that reads like a paraphrase of its",
        "paper's abstract is a question worth rewording into something a person would",
        "actually type.",
        "",
        "---",
        "",
    ]

    for index, example in enumerate(dataset.examples, start=1):
        per_config = retrieved.get(example.id, {})
        relevant = example.relevant_set

        lines += [
            f"## {index}. `{example.id}` · {example.category.value}",
            "",
            f"> **{example.query}**",
            "",
            "**Labelled relevant:**",
            "",
        ]

        for arxiv_id in example.relevant_arxiv_ids:
            lines += paper_block(arxiv_id, papers, marker="-")

        lines += [
            "**Why these were labelled** (drafted, not verified):",
            "",
            wrap(example.notes, width=88, indent="> "),
            "",
        ]

        candidates = candidate_labels(per_config, relevant, CANDIDATE_LIMIT)
        if candidates:
            lines += [
                "<details>",
                "<summary><b>Retrieved but not labelled</b> — check whether any of these "
                "should be relevant</summary>",
                "",
            ]
            for arxiv_id, ranks in candidates:
                where = ", ".join(f"{config} #{rank}" for config, rank in sorted(ranks.items()))
                lines += paper_block(arxiv_id, papers, marker=f"- _({where})_")
            lines += ["</details>", ""]

        if per_config:
            found = {
                config: ("found" if set(ranking) & relevant else "**MISSED**")
                for config, ranking in sorted(per_config.items())
            }
            lines += ["Outcome: " + " · ".join(f"{c} {v}" for c, v in found.items()), ""]

        lines += [
            "```",
            "VERDICT:",
            "REMOVE:",
            "ADD:",
            "REWORD:",
            "```",
            "",
            "---",
            "",
        ]

    lines += [
        "## When you are done",
        "",
        "Tell me the corrections and I will apply them to the dataset, flip `reviewed`",
        "to true on the examples you confirmed, change `label_source` to",
        "`human_verified`, and re-run the experiments. The re-run decides whether the",
        "dense-over-hybrid result survives review, which is what ADR-0008 says is needed",
        "before the production default changes.",
        "",
        "Partial review is still useful. Twenty reviewed questions with honest labels",
        "are worth more than thirty drafted ones, and the review status is reported per",
        "example rather than as a single flag precisely so a partial review can be",
        "recorded accurately.",
        "",
    ]

    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    """Generate the checklist."""
    dataset = load_dataset(args.dataset)
    retrieved = load_retrieved(args.results)

    referenced = dataset.referenced_papers()
    for per_config in retrieved.values():
        for ranking in per_config.values():
            referenced.update(ranking[:CANDIDATE_LIMIT])

    papers = await load_papers(referenced)

    missing = sorted(dataset.referenced_papers() - set(papers))
    if missing:
        print(f"  WARNING: {len(missing)} labelled papers are not in the corpus: {missing}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_checklist(dataset, papers, retrieved), encoding="utf-8")

    print(f"Checklist written to {args.out}")
    print(f"  {len(dataset)} questions, {len(dataset.referenced_papers())} labelled papers")
    if not retrieved:
        print(f"  NOTE: no results at {args.results}; candidate labels omitted")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a label review checklist.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
