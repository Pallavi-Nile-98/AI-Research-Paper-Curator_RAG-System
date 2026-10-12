"""Tests for the evaluation dataset and the chunk-to-paper conversion.

Two things are checked here, both of which sit between retrieval and the
reported numbers and can corrupt them silently.

``chunks_to_paper_ranking`` converts a ranked chunk list into a ranked paper
list. Every metric is computed on its output, so a bug there -- counting one
paper twice, or letting a paper's second chunk push another paper down -- would
change every score without raising anything.

The dataset's validation and review accounting decide whether a result is
presented as provisional. A dataset that reported itself as reviewed when it
was not would turn a caveat into a claim, which is the specific failure this
project is built to avoid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_curator.core.exceptions import EvaluationError
from paper_curator.evaluation.dataset import (
    EvaluationDataset,
    EvaluationExample,
    LabelSource,
    QueryCategory,
    load_dataset,
)
from paper_curator.evaluation.experiment import chunks_to_paper_ranking
from paper_curator.retrieval.models import RetrievedChunk


def make_chunk(arxiv_id: str, chunk_index: int = 0, score: float = 1.0) -> RetrievedChunk:
    """Build a RetrievedChunk with only the fields this conversion reads."""
    return RetrievedChunk(
        chunk_id=f"{arxiv_id}:{chunk_index}",
        text="text",
        score=score,
        arxiv_id=arxiv_id,
        version=1,
        title="Title",
        authors=["Author"],
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        published_at="2024-01-01",
        chunk_index=chunk_index,
    )


def make_example(**overrides: object) -> EvaluationExample:
    """Build a valid example, with fields overridable per test."""
    fields: dict[str, object] = {
        "id": "q001",
        "query": "what is BM25?",
        "category": QueryCategory.EXACT_TERMINOLOGY,
        "relevant_arxiv_ids": ["2401.00001"],
        "notes": "named in the title",
        "label_source": LabelSource.DRAFTED_FROM_ABSTRACT,
    }
    fields.update(overrides)
    return EvaluationExample(**fields)  # type: ignore[arg-type]


class TestChunksToPaperRanking:
    def test_preserves_order(self) -> None:
        chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
        assert chunks_to_paper_ranking(chunks) == ["a", "b", "c"]

    def test_deduplicates_to_best_rank(self) -> None:
        """A paper takes the position of its best chunk and appears once."""
        chunks = [make_chunk("a", 0), make_chunk("a", 1), make_chunk("b", 0)]
        assert chunks_to_paper_ranking(chunks) == ["a", "b"]

    def test_repeat_chunks_do_not_push_other_papers_down(self) -> None:
        """The property that keeps multi-source recall honest.

        Paper "b" is second among papers even though its chunk is fourth in
        the chunk list. Without collapsing, "a" would occupy three of the top
        slots and mechanically depress recall for a question needing several
        papers.
        """
        chunks = [make_chunk("a", 0), make_chunk("a", 1), make_chunk("a", 2), make_chunk("b", 0)]
        ranking = chunks_to_paper_ranking(chunks)
        assert ranking == ["a", "b"]
        assert ranking.index("b") == 1

    def test_empty(self) -> None:
        assert chunks_to_paper_ranking([]) == []

    def test_interleaved_papers(self) -> None:
        chunks = [make_chunk("a"), make_chunk("b"), make_chunk("a", 1), make_chunk("c")]
        assert chunks_to_paper_ranking(chunks) == ["a", "b", "c"]


class TestExampleValidation:
    def test_valid_example(self) -> None:
        assert make_example().relevant_set == {"2401.00001"}

    def test_rejects_empty_query(self) -> None:
        with pytest.raises(EvaluationError, match="query must not be empty"):
            make_example(query="   ")

    def test_rejects_no_relevant_papers(self) -> None:
        """An example with no relevant papers cannot discriminate anything."""
        with pytest.raises(EvaluationError, match="at least one relevant paper"):
            make_example(relevant_arxiv_ids=[])

    def test_rejects_missing_notes(self) -> None:
        """Notes are what let a reviewer check a label rather than trust it."""
        with pytest.raises(EvaluationError, match="notes explaining the relevance"):
            make_example(notes="")


class TestReviewAccounting:
    def _dataset(self, *reviewed: bool) -> EvaluationDataset:
        return EvaluationDataset(
            version="test",
            created="2026-10-12",
            examples=[
                make_example(id=f"q{index:03d}", reviewed=flag)
                for index, flag in enumerate(reviewed, start=1)
            ],
            corpus_description="test corpus",
        )

    def test_unreviewed_summary_says_so_unmistakably(self) -> None:
        summary = self._dataset(False, False).review_summary()
        assert "UNREVIEWED DRAFT" in summary
        assert "provisional" in summary

    def test_partial_review_reports_the_split(self) -> None:
        summary = self._dataset(True, False, False).review_summary()
        assert "1 human-reviewed" in summary
        assert "2 UNREVIEWED" in summary

    def test_fully_reviewed_summary_drops_the_warning(self) -> None:
        summary = self._dataset(True, True).review_summary()
        assert "all human-reviewed" in summary
        assert "UNREVIEWED" not in summary
        assert "provisional" not in summary

    def test_fully_reviewed_requires_every_example(self) -> None:
        assert not self._dataset(True, True, False).fully_reviewed
        assert self._dataset(True, True).fully_reviewed

    def test_empty_dataset_is_not_reported_as_reviewed(self) -> None:
        """Guards a vacuous truth: 0 == 0 would otherwise claim full review."""
        empty = EvaluationDataset(
            version="test", created="2026-10-12", examples=[], corpus_description=""
        )
        assert not empty.fully_reviewed


class TestLoadDataset:
    def _write(self, tmp_path: Path, payload: dict[str, object]) -> Path:
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _payload(self, examples: list[dict[str, object]]) -> dict[str, object]:
        return {
            "version": "0.1.0-test",
            "created": "2026-10-12",
            "corpus_description": "test",
            "examples": examples,
        }

    def _example(self, example_id: str = "q001") -> dict[str, object]:
        return {
            "id": example_id,
            "query": "what is BM25?",
            "category": "exact_terminology",
            "relevant_arxiv_ids": ["2401.00001"],
            "notes": "named in the title",
            "label_source": "drafted_from_abstract",
            "reviewed": False,
        }

    def test_round_trip(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, self._payload([self._example()]))
        dataset = load_dataset(path)
        assert len(dataset) == 1
        assert dataset.examples[0].category is QueryCategory.EXACT_TERMINOLOGY
        assert dataset.referenced_papers() == {"2401.00001"}

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluationError, match="not found"):
            load_dataset(tmp_path / "nope.json")

    def test_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(EvaluationError, match="not valid JSON"):
            load_dataset(path)

    def test_unknown_category_is_rejected(self, tmp_path: Path) -> None:
        bad = self._example() | {"category": "invented_category"}
        path = self._write(tmp_path, self._payload([bad]))
        with pytest.raises(EvaluationError, match="malformed example"):
            load_dataset(path)

    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        """A duplicated id would weight that query twice in every average."""
        payload = self._payload([self._example("q001"), self._example("q001")])
        path = self._write(tmp_path, payload)
        with pytest.raises(EvaluationError, match="duplicate example ids"):
            load_dataset(path)

    def test_invalid_example_fails_loudly_rather_than_being_skipped(self, tmp_path: Path) -> None:
        """Skipping a bad example would change what was measured silently."""
        bad = self._example() | {"relevant_arxiv_ids": []}
        path = self._write(tmp_path, self._payload([self._example(), bad]))
        with pytest.raises(EvaluationError):
            load_dataset(path)


class TestShippedDataset:
    """The dataset actually used to produce the recorded results."""

    PATH = Path("evaluation/datasets/retrieval_eval_v0.1-draft.json")

    @pytest.mark.skipif(not PATH.exists(), reason="dataset not present")
    def test_loads(self) -> None:
        dataset = load_dataset(self.PATH)
        assert len(dataset) == 30

    @pytest.mark.skipif(not PATH.exists(), reason="dataset not present")
    def test_every_category_is_represented(self) -> None:
        """A category with no examples would silently drop from every report."""
        dataset = load_dataset(self.PATH)
        assert set(dataset.by_category()) == {category.value for category in QueryCategory}

    @pytest.mark.skipif(not PATH.exists(), reason="dataset not present")
    def test_review_status_is_honest(self) -> None:
        """Fails the moment labels are marked reviewed without this being updated.

        Deliberately coupled: if someone flips `reviewed` to true in bulk
        without a real review, this test is the thing that should be edited
        consciously rather than a caveat that quietly disappears.
        """
        dataset = load_dataset(self.PATH)
        assert not dataset.fully_reviewed
        assert "UNREVIEWED DRAFT" in dataset.review_summary()
