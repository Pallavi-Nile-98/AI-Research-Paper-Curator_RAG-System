"""Tests for answer orchestration and the Ollama client.

The service decides when the model is called, when it is not, and what the
system is willing to stand behind afterwards. The behaviours worth guarding are
the ones that are invisible when they break:

* retrieval finds nothing and the model is **never called** -- because a model
  with no context answers from memory, fluently, and the result looks like a
  grounded answer with no citations
* a fabricated citation is **flagged**, not silently returned
* an error that will never succeed is **not retried**

Every test here uses a scripted model or a mocked transport, so none of them
touch a network or need a model pulled.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from paper_curator.core.config import Settings, get_settings
from paper_curator.generation.errors import (
    LLMTimeoutError,
    LLMUnavailableError,
    MalformedResponseError,
    ModelNotFoundError,
)
from paper_curator.generation.llm import OllamaClient, ScriptedLLM
from paper_curator.generation.models import LLMResponse, RefusalReason, TokenUsage
from paper_curator.generation.prompts import REFUSAL_TOKEN
from paper_curator.generation.service import AnswerService
from paper_curator.retrieval.models import (
    RetrievalResult,
    RetrievedChunk,
    RetrieverKind,
)

pytestmark = pytest.mark.unit


def make_chunk(arxiv_id: str, index: int = 0, text: str = "A passage.") -> RetrievedChunk:
    """Build a chunk with the fields context assembly reads."""
    return RetrievedChunk(
        chunk_id=f"{arxiv_id}:{index}",
        text=text,
        score=1.0,
        arxiv_id=arxiv_id,
        version=1,
        title=f"Paper {arxiv_id}",
        authors=["Doe", "Roe"],
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        published_at="2024-01-15",
        chunk_index=index,
        section="Introduction",
    )


class StubRetriever:
    """Returns prepared chunks without touching OpenSearch."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks if chunks is not None else [make_chunk("2401.00001")]
        self.queries: list[str] = []

    async def retrieve(
        self, query: str, *, limit: int | None = None, filters: object = None
    ) -> RetrievalResult:
        self.queries.append(query)
        return RetrievalResult(
            chunks=list(self._chunks),
            mode=RetrieverKind.HYBRID,
            query=query,
            retrieval_ms=12,
        )


def build_service(
    *, chunks: list[RetrievedChunk] | None = None, responses: list[str | LLMResponse] | None = None
) -> tuple[AnswerService, ScriptedLLM, StubRetriever]:
    """Assemble a service over a stub retriever and a scripted model."""
    settings: Settings = get_settings()
    retriever = StubRetriever(chunks)
    llm = ScriptedLLM(responses)
    service = AnswerService(retriever, llm, settings=settings)
    return service, llm, retriever


class TestNoEvidencePath:
    @pytest.mark.asyncio
    async def test_refuses_without_calling_the_model(self) -> None:
        """The most important test in this file.

        Sending an empty context and trusting the instruction to decline fails
        precisely when it matters, so the model must not be reached at all.
        """
        service, llm, _ = build_service(chunks=[])
        answer = await service.answer("What is BM25?")

        assert answer.refused
        assert answer.refusal_reason == RefusalReason.NO_EVIDENCE
        assert llm.calls == [], "model was called despite having no evidence"

    @pytest.mark.asyncio
    async def test_refusal_carries_no_citations(self) -> None:
        service, _, _ = build_service(chunks=[])
        answer = await service.answer("What is BM25?")
        assert answer.citations_used == []
        assert answer.citations_supplied == []

    @pytest.mark.asyncio
    async def test_refusal_is_not_marked_grounded_falsely(self) -> None:
        service, _, _ = build_service(chunks=[])
        answer = await service.answer("What is BM25?")
        # Nothing was cited, so nothing was invented.
        assert answer.is_grounded


class TestModelDeclined:
    @pytest.mark.asyncio
    async def test_sentinel_becomes_a_structured_refusal(self) -> None:
        service, _, _ = build_service(responses=[REFUSAL_TOKEN])
        answer = await service.answer("What is BM25?")

        assert answer.refused
        assert answer.refusal_reason == RefusalReason.MODEL_DECLINED
        assert REFUSAL_TOKEN not in answer.text, "raw sentinel leaked to the user"

    @pytest.mark.asyncio
    async def test_declined_answer_keeps_the_sources_it_was_given(self) -> None:
        """So a reviewer can see what the model declined to use."""
        service, _, _ = build_service(responses=[REFUSAL_TOKEN])
        answer = await service.answer("What is BM25?")
        assert len(answer.citations_supplied) == 1


class TestGroundedAnswer:
    @pytest.mark.asyncio
    async def test_valid_citation_is_recorded_as_used(self) -> None:
        service, _, _ = build_service(responses=["BM25 is a ranking function [P1]."])
        answer = await service.answer("What is BM25?")

        assert not answer.refused
        assert [c.label for c in answer.citations_used] == ["P1"]
        assert answer.is_grounded
        assert not answer.has_problems

    @pytest.mark.asyncio
    async def test_records_prompt_version_and_model(self) -> None:
        """Without these, a quality regression cannot be attributed."""
        service, _, _ = build_service(responses=["An answer [P1]."])
        answer = await service.answer("What is BM25?")
        assert answer.prompt_version
        assert answer.model == "scripted"

    @pytest.mark.asyncio
    async def test_reports_latency_split(self) -> None:
        service, _, _ = build_service(responses=["An answer [P1]."])
        answer = await service.answer("What is BM25?")
        assert answer.retrieval_ms >= 0
        assert answer.total_ms == answer.retrieval_ms + answer.generation_ms

    @pytest.mark.asyncio
    async def test_two_papers_get_distinct_labels(self) -> None:
        chunks = [make_chunk("2401.00001"), make_chunk("2401.00002")]
        service, _, _ = build_service(chunks=chunks, responses=["Both agree [P1][P2]."])
        answer = await service.answer("What is BM25?")
        assert [c.label for c in answer.citations_used] == ["P1", "P2"]
        assert {c.arxiv_id for c in answer.citations_used} == {"2401.00001", "2401.00002"}


class TestInventedCitations:
    @pytest.mark.asyncio
    async def test_invented_label_is_flagged(self) -> None:
        """The silent failure: the sentence reads perfectly."""
        service, _, _ = build_service(responses=["A fabricated claim [P9]."])
        answer = await service.answer("What is BM25?")

        assert answer.invented_labels == ["P9"]
        assert not answer.is_grounded
        assert answer.has_problems

    @pytest.mark.asyncio
    async def test_answer_text_is_preserved_not_silently_stripped(self) -> None:
        """Stripping would destroy the evidence that the model fabricated."""
        service, _, _ = build_service(responses=["A fabricated claim [P9]."])
        answer = await service.answer("What is BM25?")
        assert "[P9]" in answer.text

    @pytest.mark.asyncio
    async def test_mixed_answer_reports_both_sides(self) -> None:
        service, _, _ = build_service(responses=["Real [P1] and invented [P4]."])
        answer = await service.answer("What is BM25?")
        assert [c.label for c in answer.citations_used] == ["P1"]
        assert answer.invented_labels == ["P4"]

    @pytest.mark.asyncio
    async def test_malformed_is_distinct_from_invented(self) -> None:
        service, _, _ = build_service(responses=["A claim [1]."])
        answer = await service.answer("What is BM25?")
        assert answer.invented_labels == []
        assert answer.malformed_labels == ["1"]


class TestUncitedSources:
    @pytest.mark.asyncio
    async def test_unused_source_is_reported_but_not_an_error(self) -> None:
        chunks = [make_chunk("2401.00001"), make_chunk("2401.00002")]
        service, _, _ = build_service(chunks=chunks, responses=["Only one [P1]."])
        answer = await service.answer("What is BM25?")

        assert [c.label for c in answer.uncited_sources] == ["P2"]
        assert answer.is_grounded
        assert not answer.has_problems


class TestTokenLimit:
    @pytest.mark.asyncio
    async def test_truncated_output_is_flagged(self) -> None:
        """A cut-off answer can lose its final citation mid-sentence."""
        cut_off = LLMResponse(
            text="A long answer that stops abruptly [P1] and then",
            model="scripted",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=1024),
            finish_reason="length",
        )
        service, _, _ = build_service(responses=[cut_off])
        answer = await service.answer("What is BM25?")

        assert answer.hit_token_limit
        assert answer.has_problems


class TestInputValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("question", ["", "   ", "\n\t"])
    async def test_empty_question_rejected(self, question: str) -> None:
        service, _, _ = build_service()
        with pytest.raises(ValueError, match="must not be empty"):
            await service.answer(question)


class TestOllamaClientErrors:
    """Error classification, which decides what gets retried."""

    def _client(self) -> OllamaClient:
        return OllamaClient(get_settings())

    @pytest.mark.asyncio
    @respx.mock
    async def test_missing_model_is_not_retryable(self) -> None:
        """A 404 fails identically forever; retrying only delays the message."""
        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(404, json={"error": "model not found"})
        )
        client = self._client()
        with pytest.raises(ModelNotFoundError):
            await client.complete([{"role": "user", "content": "hi"}])

        assert route.call_count == 1, "a permanent failure was retried"
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_server_error_is_retried(self) -> None:
        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(503, text="overloaded")
        )
        client = self._client()
        with pytest.raises(LLMUnavailableError):
            await client.complete([{"role": "user", "content": "hi"}])

        assert route.call_count > 1, "a transient failure was not retried"
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_timeout_is_classified_and_retried(self) -> None:
        route = respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ReadTimeout("too slow")
        )
        client = self._client()
        with pytest.raises(LLMTimeoutError):
            await client.complete([{"role": "user", "content": "hi"}])

        assert route.call_count > 1
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_response_without_content_is_malformed(self) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"model": "llama3.2:3b", "done": True})
        )
        client = self._client()
        with pytest.raises(MalformedResponseError):
            await client.complete([{"role": "user", "content": "hi"}])
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_response_is_parsed(self) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "llama3.2:3b",
                    "message": {"role": "assistant", "content": "An answer [P1]."},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 850,
                    "eval_count": 42,
                },
            )
        )
        client = self._client()
        result = await client.complete([{"role": "user", "content": "hi"}])

        assert result.text == "An answer [P1]."
        assert result.usage.prompt_tokens == 850
        assert result.usage.completion_tokens == 42
        assert result.usage.total_tokens == 892
        assert not result.hit_token_limit
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_length_finish_reason_is_surfaced(self) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "llama3.2:3b",
                    "message": {"role": "assistant", "content": "Cut off"},
                    "done_reason": "length",
                },
            )
        )
        client = self._client()
        result = await client.complete([{"role": "user", "content": "hi"}])
        assert result.hit_token_limit
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_false_when_model_absent(self) -> None:
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "mistral:7b"}]})
        )
        client = self._client()
        assert not await client.health()
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_true_when_model_present(self) -> None:
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})
        )
        client = self._client()
        assert await client.health()
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_false_when_server_unreachable(self) -> None:
        """Returns False rather than raising: it decides whether to start work."""
        respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("refused"))
        client = self._client()
        assert not await client.health()
        await client.close()
