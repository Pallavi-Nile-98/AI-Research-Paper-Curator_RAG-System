"""Turning a question into a grounded, cited answer.

The orchestration layer: retrieve, optionally re-rank, assemble context,
generate, then verify what came back. Business logic lives here rather than in
an API route, so the same path can be exercised from a test, a script or an
HTTP handler without any of them re-implementing it.

Two control-flow decisions carry most of the weight.

**No evidence means no model call.** When retrieval returns nothing the service
refuses immediately, without asking the model. The alternative -- send the
question with an empty context and trust the instruction to decline -- fails
exactly when it matters: a model with nothing to ground against answers from
its parameters, fluently, and the result is indistinguishable from a grounded
answer apart from having no citations.

**Verification is unconditional.** Every answer is checked against the sources
it was given, including answers that look fine. The check is cheap and the
failure it catches is invisible without it.
"""

from __future__ import annotations

import time
from typing import Protocol

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.logging import get_logger
from paper_curator.generation import citations as citation_tools
from paper_curator.generation import prompts
from paper_curator.generation.llm import LLMClient
from paper_curator.generation.models import GeneratedAnswer, RefusalReason
from paper_curator.retrieval.context import AssembledContext, ContextBuilder
from paper_curator.retrieval.models import RetrievalFilters, RetrievalResult
from paper_curator.retrieval.reranking import RerankingService

logger = get_logger(__name__)

NO_EVIDENCE_MESSAGE = (
    "I could not find anything in the indexed papers that addresses this "
    "question. Try rephrasing it, or it may be outside what this corpus covers."
)

MODEL_DECLINED_MESSAGE = (
    "The papers I found do not contain enough information to answer this "
    "question reliably, so I would rather not guess."
)


class Retriever(Protocol):
    """The retrieval surface this service depends on.

    A protocol rather than a concrete type, so the service works with the
    keyword, vector and hybrid retrievers alike -- and so a test can supply a
    stub without constructing an OpenSearch client.
    """

    async def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        """Return ranked chunks for a query."""
        ...


class AnswerService:
    """Answers questions from the indexed corpus."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        *,
        settings: Settings | None = None,
        reranker: RerankingService | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._retriever = retriever
        self._llm = llm
        self._reranker = reranker
        self._context = context_builder or ContextBuilder(self._settings)

    async def _gather_context(
        self, question: str, filters: RetrievalFilters | None
    ) -> tuple[AssembledContext, int]:
        """Retrieve, optionally re-rank, and assemble prompt-ready context."""
        started = time.perf_counter()

        retrieved = await self._retriever.retrieve(
            question,
            limit=self._settings.retrieval.candidate_pool_size,
            filters=filters,
        )
        chunks = list(retrieved.chunks)

        if self._reranker is not None and chunks:
            outcome = await self._reranker.rerank(
                question, chunks, top_k=self._settings.retrieval.top_k
            )
            chunks = list(outcome.chunks)
        else:
            chunks = chunks[: self._settings.retrieval.top_k]

        context = self._context.build(chunks)
        return context, int((time.perf_counter() - started) * 1000)

    def _refuse(
        self, question: str, reason: str, message: str, *, retrieval_ms: int
    ) -> GeneratedAnswer:
        """Build a refusal, which is a legitimate outcome rather than a failure."""
        logger.info("answer_refused", reason=reason, question=question[:120])
        return GeneratedAnswer(
            text=message,
            question=question,
            refused=True,
            refusal_reason=reason,
            prompt_version=prompts.PROMPT_VERSION,
            retrieval_ms=retrieval_ms,
        )

    async def answer(
        self, question: str, *, filters: RetrievalFilters | None = None
    ) -> GeneratedAnswer:
        """Answer a question from the corpus.

        Raises:
            ValueError: The question is empty.

        """
        if not question.strip():
            msg = "question must not be empty"
            raise ValueError(msg)

        context, retrieval_ms = await self._gather_context(question, filters)

        if context.is_empty:
            return self._refuse(
                question,
                RefusalReason.NO_EVIDENCE,
                NO_EVIDENCE_MESSAGE,
                retrieval_ms=retrieval_ms,
            )

        messages = prompts.build_messages(question, context)
        started = time.perf_counter()
        response = await self._llm.complete(messages)
        generation_ms = int((time.perf_counter() - started) * 1000)

        if prompts.is_refusal(response.text):
            refusal = self._refuse(
                question,
                RefusalReason.MODEL_DECLINED,
                MODEL_DECLINED_MESSAGE,
                retrieval_ms=retrieval_ms,
            )
            # Rebuilt rather than mutated: the refusal helper does not know
            # about the generation that produced it, and the sources are worth
            # keeping so a reviewer can see what the model declined to use.
            return GeneratedAnswer(
                text=refusal.text,
                question=question,
                citations_supplied=context.citations,
                refused=True,
                refusal_reason=RefusalReason.MODEL_DECLINED,
                model=response.model,
                prompt_version=prompts.PROMPT_VERSION,
                usage=response.usage,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                truncated_context=context.truncated,
            )

        used, invented, malformed = citation_tools.classify(response.text, context.citations)

        answer = GeneratedAnswer(
            text=response.text.strip(),
            question=question,
            citations_used=used,
            citations_supplied=context.citations,
            invented_labels=invented,
            malformed_labels=malformed,
            model=response.model,
            prompt_version=prompts.PROMPT_VERSION,
            usage=response.usage,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            truncated_context=context.truncated,
            hit_token_limit=response.hit_token_limit,
        )

        if invented:
            # Logged at warning because it is the failure this system is built
            # to prevent, and a silent one: the answer reads correctly.
            logger.warning(
                "answer_cited_unsupplied_labels",
                invented=invented,
                supplied=sorted(context.valid_labels),
                question=question[:120],
            )

        logger.info("answer_generated", summary=answer.summary())
        return answer
