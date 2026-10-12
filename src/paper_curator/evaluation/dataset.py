"""Loading and validating the retrieval evaluation dataset.

The dataset is the measurement instrument. If its labels are wrong, every
number downstream is wrong in a way no amount of careful experimental
methodology can detect -- so the structure here is built to make the status of
a label impossible to lose track of.

Two fields carry that weight:

``reviewed`` -- whether a human has checked this example. An unreviewed label
is a hypothesis about relevance, not a fact about it. :meth:`EvaluationDataset.review_summary`
reports the split, and the experiment report states it, so no result is ever
presented as though its labels were verified when they were not.

``label_source`` -- how the label was produced, so a reader can judge it
without taking anyone's word. "Drafted from the abstract" and "confirmed by
reading the method section" are different levels of evidence.

Relevance is recorded at **paper** level. Chunk-level labels would mean a human
judging every one of several thousand chunks against every query; the resulting
labels would not be trustworthy, and untrustworthy labels are worse than none.
Section hints are optional and advisory -- they help a reviewer find the
supporting passage, and no metric depends on them.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from paper_curator.core.exceptions import EvaluationError


class QueryCategory(StrEnum):
    """The kind of question, which is what the comparison turns on.

    Reported per category as well as overall, because the aggregate hides the
    thing the experiment exists to show. BM25 and dense retrieval are expected
    to diverge on specific categories; averaging across all of them can leave
    two very different systems looking equivalent.
    """

    EXACT_TERMINOLOGY = "exact_terminology"
    """Names a specific method, metric or system. Favours lexical matching."""

    CONCEPTUAL = "conceptual"
    """Asks about an idea in the user's own words. Favours semantic matching."""

    METHOD_COMPARISON = "method_comparison"
    """Contrasts approaches; usually needs more than one paper."""

    PAPER_SPECIFIC = "paper_specific"
    """Targets a known paper, author or identifier."""

    MULTI_SOURCE = "multi_source"
    """Cannot be answered from a single paper."""

    VOCABULARY_MISMATCH = "vocabulary_mismatch"
    """Deliberately avoids the corpus's own wording.

    The sharpest test of the ADR-0004 hypothesis: exactly where BM25 should
    fail and dense retrieval should not.
    """


class LabelSource(StrEnum):
    """How a relevance label was arrived at."""

    DRAFTED_FROM_ABSTRACT = "drafted_from_abstract"
    """Proposed by reading the paper's abstract. Unverified."""

    DRAFTED_FROM_FULL_TEXT = "drafted_from_full_text"
    """Proposed by reading indexed body text. Unverified."""

    HUMAN_VERIFIED = "human_verified"
    """A person read the paper and confirmed it answers the query."""

    HUMAN_AUTHORED = "human_authored"
    """A person wrote both the question and its labels."""


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """One labelled query."""

    id: str
    query: str
    category: QueryCategory
    relevant_arxiv_ids: list[str]
    notes: str
    label_source: LabelSource
    reviewed: bool = False
    relevant_sections: list[str] = field(default_factory=list)
    """Advisory hints about where the answer sits. No metric reads these."""

    @property
    def relevant_set(self) -> set[str]:
        return set(self.relevant_arxiv_ids)

    def __post_init__(self) -> None:
        if not self.query.strip():
            msg = f"example {self.id}: query must not be empty"
            raise EvaluationError(msg)
        if not self.relevant_arxiv_ids:
            # An example with no relevant papers cannot distinguish a system
            # that retrieves nothing from one that retrieves everything.
            msg = f"example {self.id}: at least one relevant paper is required"
            raise EvaluationError(msg)
        if not self.notes.strip():
            # The note is what lets a reviewer check the label rather than
            # trust it, so it is required rather than optional.
            msg = f"example {self.id}: notes explaining the relevance are required"
            raise EvaluationError(msg)


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """A versioned set of labelled queries."""

    version: str
    created: str
    examples: list[EvaluationExample]
    corpus_description: str
    notes: str = ""

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def reviewed_count(self) -> int:
        return sum(1 for example in self.examples if example.reviewed)

    @property
    def fully_reviewed(self) -> bool:
        """True only when there are examples and every one has been checked.

        The emptiness guard is not pedantry. Without it ``0 == 0`` makes an
        empty dataset report itself as fully reviewed, and the provisional
        warning disappears from the report at exactly the moment the data is
        most broken -- a caveat that vanishes when something goes wrong is
        worse than no caveat.
        """
        return bool(self.examples) and self.reviewed_count == len(self.examples)

    def by_category(self) -> dict[str, int]:
        """How many examples fall in each category."""
        return dict(Counter(example.category.value for example in self.examples))

    def review_summary(self) -> str:
        """One line stating the review status, for inclusion in any report.

        Written to be impossible to read as an endorsement when it is not one.
        """
        if self.fully_reviewed:
            return f"{len(self)} examples, all human-reviewed"
        return (
            f"{len(self)} examples, {self.reviewed_count} human-reviewed, "
            f"{len(self) - self.reviewed_count} UNREVIEWED DRAFT -- "
            "results computed against unreviewed labels are provisional"
        )

    def referenced_papers(self) -> set[str]:
        """Every arXiv ID any example marks relevant."""
        return {arxiv_id for example in self.examples for arxiv_id in example.relevant_arxiv_ids}


def load_dataset(path: Path) -> EvaluationDataset:
    """Load a dataset from JSON.

    Raises:
        EvaluationError: The file is missing, malformed, or an example is
            invalid. Failing loudly matters here: silently skipping a bad
            example would change what was measured without changing the
            reported dataset size.

    """
    if not path.exists():
        msg = f"evaluation dataset not found: {path}"
        raise EvaluationError(msg, context={"path": str(path)})

    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"evaluation dataset is not valid JSON: {exc}"
        raise EvaluationError(msg, context={"path": str(path)}) from exc

    try:
        examples = [
            EvaluationExample(
                id=item["id"],
                query=item["query"],
                category=QueryCategory(item["category"]),
                relevant_arxiv_ids=list(item["relevant_arxiv_ids"]),
                relevant_sections=list(item.get("relevant_sections", [])),
                notes=item["notes"],
                label_source=LabelSource(item["label_source"]),
                reviewed=bool(item.get("reviewed", False)),
            )
            for item in raw["examples"]
        ]
    except (KeyError, ValueError) as exc:
        msg = f"evaluation dataset has a malformed example: {exc}"
        raise EvaluationError(msg, context={"path": str(path)}) from exc

    ids = [example.id for example in examples]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        # A duplicated id would be silently counted twice, weighting those
        # queries more heavily than the rest.
        msg = f"evaluation dataset has duplicate example ids: {duplicates}"
        raise EvaluationError(msg, context={"path": str(path)})

    return EvaluationDataset(
        version=raw["version"],
        created=raw["created"],
        examples=examples,
        corpus_description=raw.get("corpus_description", ""),
        notes=raw.get("notes", ""),
    )
