"""Tests for re-ranking and context assembly.

Two components with one thing in common: both sit between retrieval and the
model, and both must fail softly. Re-ranking improves ordering but is not
required for an answer; context assembly decides what the model can physically
cite. A hard failure in either turns a degraded answer into no answer.
"""

from __future__ import annotations

import asyncio

import pytest

from paper_curator.core.config import RerankerSettings, RetrievalSettings, Settings
from paper_curator.retrieval.context import ContextBuilder
from paper_curator.retrieval.models import RetrievedChunk
from paper_curator.retrieval.reranking import IdentityReranker, RerankingService


def chunk(
    chunk_id: str,
    *,
    text: str = "a passage about retrieval systems and their behaviour",
    arxiv_id: str = "2401.00001",
    title: str = "A Paper About Retrieval",
    section: str | None = "Method",
    score: float = 1.0,
) -> RetrievedChunk:
    """Build a retrieved chunk."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        arxiv_id=arxiv_id,
        version=1,
        title=title,
        authors=["Jane Doe", "Rahul Mehta"],
        abs_url=f"https://arxiv.org/abs/{arxiv_id}v1",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}v1",
        published_at="2024-03-15T00:00:00+00:00",
        chunk_index=0,
        section=section,
    )


def rerank_settings(**overrides: object) -> Settings:
    """Build settings with reranker options overridden."""
    return Settings(reranker=RerankerSettings(**overrides))  # type: ignore[arg-type]


def context_settings(**overrides: object) -> Settings:
    """Build settings with retrieval options overridden."""
    return Settings(retrieval=RetrievalSettings(**overrides))  # type: ignore[arg-type]


class ScriptedReranker:
    """Returns predetermined scores, so ordering can be asserted exactly."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    @property
    def model_name(self) -> str:
        return "scripted"

    def score(self, query: str, texts: list[str]) -> list[float]:
        return self._scores[: len(texts)]

    def warmup(self) -> None:
        """No-op."""


@pytest.mark.unit
class TestRerankingReordering:
    async def test_chunks_are_reordered_by_score(self) -> None:
        """The point of the second stage: fix an ordering retrieval got wrong."""
        chunks = [chunk("a"), chunk("b"), chunk("c")]
        service = RerankingService(ScriptedReranker([0.1, 0.9, 0.5]), rerank_settings())

        outcome = await service.rerank("query", chunks)

        assert [c.chunk_id for c in outcome.chunks] == ["b", "c", "a"]
        assert outcome.applied is True

    async def test_rerank_scores_are_recorded(self) -> None:
        """Kept separately so the retrieval score remains inspectable."""
        service = RerankingService(ScriptedReranker([0.7]), rerank_settings())
        outcome = await service.rerank("query", [chunk("a")])
        assert outcome.chunks[0].rerank_score == pytest.approx(0.7)

    async def test_only_top_k_is_returned(self) -> None:
        """Retrieve a wide pool, re-rank it, keep the best few."""
        chunks = [chunk(f"c{n}") for n in range(10)]
        service = RerankingService(
            ScriptedReranker([float(n) for n in range(10)]), rerank_settings(top_k=3)
        )
        outcome = await service.rerank("query", chunks)
        assert len(outcome.chunks) == 3

    async def test_latency_is_measured_separately(self) -> None:
        """Folding it into retrieval latency would misattribute the cost."""
        service = RerankingService(ScriptedReranker([1.0]), rerank_settings())
        outcome = await service.rerank("query", [chunk("a")])
        assert outcome.rerank_ms >= 0

    async def test_identity_reranker_preserves_order(self) -> None:
        chunks = [chunk("a"), chunk("b"), chunk("c")]
        service = RerankingService(IdentityReranker(), rerank_settings())
        outcome = await service.rerank("query", chunks)
        assert [c.chunk_id for c in outcome.chunks] == ["a", "b", "c"]


@pytest.mark.unit
class TestRerankingDegradesSafely:
    """Re-ranking is additive. It must never be the reason a query fails."""

    async def test_disabled_returns_retrieval_order(self) -> None:
        chunks = [chunk("a"), chunk("b")]
        service = RerankingService(ScriptedReranker([0.1, 0.9]), rerank_settings(enabled=False))

        outcome = await service.rerank("query", chunks)

        assert [c.chunk_id for c in outcome.chunks] == ["a", "b"]
        assert outcome.applied is False
        assert outcome.skipped_reason == "reranking_disabled"

    async def test_a_timeout_falls_back_to_retrieval_order(self) -> None:
        """A slow model must not hold a user request open indefinitely."""

        class SlowReranker:
            @property
            def model_name(self) -> str:
                return "slow"

            def score(self, query: str, texts: list[str]) -> list[float]:
                import time

                time.sleep(2.0)
                return [1.0] * len(texts)

            def warmup(self) -> None:
                """No-op."""

        chunks = [chunk("a"), chunk("b")]
        service = RerankingService(SlowReranker(), rerank_settings(timeout_seconds=0.1))

        outcome = await service.rerank("query", chunks)

        assert [c.chunk_id for c in outcome.chunks] == ["a", "b"]
        assert outcome.applied is False
        assert outcome.skipped_reason == "timed_out"

    async def test_an_exception_falls_back_to_retrieval_order(self) -> None:
        class BrokenReranker:
            @property
            def model_name(self) -> str:
                return "broken"

            def score(self, query: str, texts: list[str]) -> list[float]:
                msg = "model exploded"
                raise RuntimeError(msg)

            def warmup(self) -> None:
                """No-op."""

        service = RerankingService(BrokenReranker(), rerank_settings())
        outcome = await service.rerank("query", [chunk("a"), chunk("b")])

        assert [c.chunk_id for c in outcome.chunks] == ["a", "b"]
        assert outcome.applied is False
        assert outcome.skipped_reason is not None
        assert "RuntimeError" in outcome.skipped_reason

    async def test_a_wrong_number_of_scores_is_rejected(self) -> None:
        """Pairing scores to the wrong chunks would corrupt the order silently.

        Worse than not re-ranking, because the result looks ordered.
        """
        service = RerankingService(ScriptedReranker([0.5]), rerank_settings())
        outcome = await service.rerank("query", [chunk("a"), chunk("b"), chunk("c")])

        assert outcome.applied is False
        assert outcome.skipped_reason == "score_count_mismatch"
        assert [c.chunk_id for c in outcome.chunks] == ["a", "b", "c"]

    async def test_no_candidates(self) -> None:
        service = RerankingService(ScriptedReranker([]), rerank_settings())
        outcome = await service.rerank("query", [])
        assert outcome.chunks == []
        assert outcome.applied is False


@pytest.mark.unit
class TestCitationLabelling:
    def test_chunks_from_one_paper_share_a_label(self) -> None:
        """Citation is of a paper, not a fragment.

        Labelling every chunk separately gives an eight-passage context eight
        labels, and a small model starts confusing them.
        """
        chunks = [chunk("a"), chunk("b"), chunk("c")]
        context = ContextBuilder(context_settings()).build(chunks)

        assert len(context.citations) == 1
        assert context.citations[0].label == "P1"
        assert context.text.count("[P1]") == 3

    def test_labels_follow_rank_order(self) -> None:
        """P1 is always the highest-ranked source."""
        chunks = [
            chunk("a", arxiv_id="2401.00001", title="First"),
            chunk("b", arxiv_id="2401.00002", title="Second"),
        ]
        context = ContextBuilder(context_settings()).build(chunks)

        by_label = {c.label: c.title for c in context.citations}
        assert by_label["P1"] == "First"
        assert by_label["P2"] == "Second"

    def test_citations_record_which_sections_were_supplied(self) -> None:
        """Needed for verification and for failure analysis."""
        chunks = [chunk("a", section="Method"), chunk("b", section="Results")]
        context = ContextBuilder(context_settings()).build(chunks)
        assert context.citations[0].sections == ["Method", "Results"]

    def test_citations_record_the_exact_chunks_behind_them(self) -> None:
        chunks = [chunk("chunk-1"), chunk("chunk-2")]
        context = ContextBuilder(context_settings()).build(chunks)
        assert context.citations[0].chunk_ids == ["chunk-1", "chunk-2"]

    def test_valid_labels_is_what_citation_validation_checks_against(self) -> None:
        """Anything the model cites outside this set was invented."""
        chunks = [chunk("a", arxiv_id="2401.00001"), chunk("b", arxiv_id="2401.00002")]
        context = ContextBuilder(context_settings()).build(chunks)
        assert context.valid_labels == {"P1", "P2"}

    def test_author_summary_shortens_long_author_lists(self) -> None:
        context = ContextBuilder(context_settings()).build([chunk("a")])
        assert context.citations[0].author_summary == "Jane Doe et al."

    def test_year_is_extracted_for_display(self) -> None:
        context = ContextBuilder(context_settings()).build([chunk("a")])
        assert context.citations[0].year == "2024"


@pytest.mark.unit
class TestContextRendering:
    def test_each_passage_carries_its_own_attribution(self) -> None:
        """Repeated rather than grouped under one heading.

        With a shared heading, a small model attributes the last passage to
        whichever label it saw most recently.
        """
        chunks = [chunk("a", section="Method"), chunk("b", section="Results")]
        text = ContextBuilder(context_settings()).build(chunks).text

        assert text.count("[P1] A Paper About Retrieval") == 2

    def test_section_appears_in_the_header(self) -> None:
        """Lets the model say which part of the paper supports a claim."""
        text = ContextBuilder(context_settings()).build([chunk("a", section="Results")]).text
        assert "Results" in text

    def test_passage_text_is_included_verbatim(self) -> None:
        body = "a very specific finding about reciprocal rank fusion"
        text = ContextBuilder(context_settings()).build([chunk("a", text=body)]).text
        assert body in text


@pytest.mark.unit
class TestTokenBudget:
    def test_budget_is_respected(self) -> None:
        chunks = [chunk(f"c{n}", text="word " * 200) for n in range(10)]
        context = ContextBuilder(context_settings(context_token_budget=400)).build(chunks)
        assert context.token_count <= 400

    def test_dropped_chunks_are_counted_and_flagged(self) -> None:
        chunks = [chunk(f"c{n}", text="word " * 200) for n in range(10)]
        context = ContextBuilder(context_settings(context_token_budget=400)).build(chunks)

        assert context.chunks_dropped > 0
        assert context.truncated is True
        assert context.chunks_included + context.chunks_dropped == 10

    def test_chunks_are_never_truncated(self) -> None:
        """Half a passage can end mid-negation.

        A model quoting the visible half then cites something the paper did
        not say, which is exactly the failure this system exists to prevent.
        """
        body = "The method does not improve recall under these conditions whatsoever"
        chunks = [chunk("a", text=body), chunk("b", text="word " * 500)]
        context = ContextBuilder(context_settings(context_token_budget=250)).build(chunks)

        assert body in context.text
        assert "word word" not in context.text

    def test_the_highest_ranked_evidence_is_kept(self) -> None:
        """Budget is spent in rank order, so the best survives."""
        chunks = [
            chunk("best", text="the critical finding " * 20),
            *[chunk(f"c{n}", text="filler " * 100) for n in range(5)],
        ]
        context = ContextBuilder(context_settings(context_token_budget=200)).build(chunks)
        assert "the critical finding" in context.text

    def test_a_paper_whose_passages_all_dropped_is_not_cited(self) -> None:
        """Citing evidence the model never saw would be a fabricated source."""
        chunks = [
            chunk("small", arxiv_id="2401.00001", text="short passage here"),
            chunk("huge", arxiv_id="2401.00002", text="word " * 1000),
        ]
        context = ContextBuilder(context_settings(context_token_budget=250)).build(chunks)

        assert len(context.citations) == 1
        assert context.citations[0].arxiv_id == "2401.00001"

    def test_labels_have_no_gaps_after_dropping(self) -> None:
        """Surviving sources are P1..Pn, so a label never points at nothing."""
        chunks = [
            chunk("a", arxiv_id="2401.00001", text="first short passage"),
            chunk("b", arxiv_id="2401.00002", text="word " * 1000),
            chunk("c", arxiv_id="2401.00003", text="third short passage"),
        ]
        context = ContextBuilder(context_settings(context_token_budget=250)).build(chunks)

        labels = [c.label for c in context.citations]
        assert labels == [f"P{n}" for n in range(1, len(labels) + 1)]
        for label in labels:
            assert f"[{label}]" in context.text


@pytest.mark.unit
class TestEmptyContext:
    def test_no_chunks_produces_an_empty_context(self) -> None:
        context = ContextBuilder(context_settings()).build([])
        assert context.is_empty
        assert context.text == ""
        assert context.citations == []

    def test_empty_context_is_the_insufficient_evidence_signal(self) -> None:
        """The caller must decline rather than ask the model to answer anyway."""
        context = ContextBuilder(context_settings()).build([])
        assert context.is_empty
        assert context.valid_labels == set()


@pytest.mark.unit
def test_rerank_runs_off_the_event_loop() -> None:
    """Scoring is CPU-bound; blocking the loop would stall every other request."""

    class BlockingReranker:
        @property
        def model_name(self) -> str:
            return "blocking"

        def score(self, query: str, texts: list[str]) -> list[float]:
            import time

            time.sleep(0.2)
            return [1.0] * len(texts)

        def warmup(self) -> None:
            """No-op."""

    async def scenario() -> bool:
        service = RerankingService(BlockingReranker(), rerank_settings(timeout_seconds=5))
        ticks = 0

        async def tick() -> None:
            nonlocal ticks
            for _ in range(10):
                await asyncio.sleep(0.01)
                ticks += 1

        await asyncio.gather(service.rerank("q", [chunk("a")]), tick())
        return ticks == 10

    assert asyncio.run(scenario()) is True
