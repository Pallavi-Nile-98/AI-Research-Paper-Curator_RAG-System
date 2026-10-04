"""Cross-encoder re-ranking of the retrieved candidate pool.

Retrieval and re-ranking answer the same question with opposite tradeoffs.

A **bi-encoder** -- what vector retrieval uses -- embeds the query and every
chunk *separately*, so chunk vectors are computed once at index time and a
query is one vector comparison against millions. Fast, and necessarily
approximate: the model never sees the query and the chunk together, so it
cannot notice that a passage answers *this particular* question.

A **cross-encoder** reads the query and one chunk *as a single input* and
scores the pair directly. Far more accurate, and far too slow to run over a
corpus: it is one model forward pass per candidate, so scoring a million chunks
per query is not a thing anyone does.

Hence the two-stage shape: retrieve a wide pool cheaply, then re-score the
handful of survivors expensively. Recall comes from the first stage, precision
from the second. A relevant chunk the retriever never returned cannot be
rescued by re-ranking -- which is why the candidate pool is deliberately much
larger than the final top-k.

Two properties are load-bearing here:

**It must degrade rather than fail.** Re-ranking improves ordering; it is not
required for an answer. If the model is missing, slow, or errors, the
retrieval order is already a reasonable answer and is returned unchanged.
Turning an optional quality improvement into a hard dependency would make the
whole system fail for a component that is meant to be additive.

**It must be measured separately.** Cross-encoder inference on a CPU is
significant, and folding that cost into retrieval latency would misattribute
it. Phase 2 compares configurations with and without re-ranking precisely to
weigh that cost against whatever quality it buys.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.exceptions import ConfigurationError
from paper_curator.core.logging import get_logger
from paper_curator.retrieval.models import RetrievedChunk

logger = get_logger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Rescores query-chunk pairs."""

    @property
    def model_name(self) -> str:
        """Identifier of the scoring model, recorded with every experiment."""
        ...

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return a relevance score per text, higher meaning more relevant.

        Scores are comparable only within one call. Cross-encoder outputs are
        unbounded logits, not probabilities, so a score of 4.2 means something
        relative to the other candidates for this query and nothing at all
        across queries.
        """
        ...

    def warmup(self) -> None:
        """Load the model now rather than inside a timed operation."""
        ...


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    """Re-ranked chunks and what it cost."""

    chunks: list[RetrievedChunk]
    rerank_ms: int
    applied: bool
    """False when re-ranking was skipped or failed and the input was returned."""

    skipped_reason: str | None = None
    candidates_scored: int = 0


class CrossEncoderReranker:
    """Scores query-chunk pairs with a local cross-encoder.

    The model is loaded lazily, so constructing this is cheap and a process
    that never re-ranks never pays the load cost.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        config = (settings or get_settings()).reranker
        self._model_name = config.model_name
        self._device = config.device
        self._batch_size = config.batch_size
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            msg = (
                "sentence-transformers is not installed, so re-ranking is "
                'unavailable. Install the optional extra with: pip install -e ".[ml]"'
            )
            raise ConfigurationError(msg) from exc

        logger.info("reranker_loading", model=self._model_name, device=self._device)
        self._model = CrossEncoder(self._model_name, device=self._device)
        logger.info("reranker_loaded", model=self._model_name)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score every text against the query. Blocking and CPU-bound."""
        if not texts:
            return []
        model = self._load()
        scores = model.predict(
            [(query, text) for text in texts],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    def warmup(self) -> None:
        """Load the model and run one pass, so timing excludes both."""
        self._load()
        self.score("warmup query", ["warmup passage"])


class IdentityReranker:
    """Returns the input order unchanged.

    Used in tests and as the configured no-op, so the pipeline has one code
    path whether or not re-ranking is enabled. Scores descend so the
    contract -- higher is better -- still holds.
    """

    @property
    def model_name(self) -> str:
        return "identity"

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(len(texts) - index) for index in range(len(texts))]

    def warmup(self) -> None:
        """No-op: there is nothing to load."""


class RerankingService:
    """Applies a reranker with a timeout and a safe fallback."""

    def __init__(
        self,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._config = resolved.reranker
        self._reranker = reranker or CrossEncoderReranker(resolved)

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int | None = None
    ) -> RerankOutcome:
        """Rescore ``chunks`` and return the best ``top_k``.

        Never raises for a re-ranking problem. A timeout, a missing model or an
        inference error all return the input order with ``applied=False`` and a
        reason, because an unordered-but-present answer beats no answer.
        """
        limit = top_k or self._config.top_k

        if not self._config.enabled:
            return RerankOutcome(
                chunks=chunks[:limit],
                rerank_ms=0,
                applied=False,
                skipped_reason="reranking_disabled",
            )

        if not chunks:
            return RerankOutcome(
                chunks=[], rerank_ms=0, applied=False, skipped_reason="no_candidates"
            )

        started = time.perf_counter()
        try:
            scores = await asyncio.wait_for(
                # Blocking and CPU-bound; keep it off the event loop.
                asyncio.to_thread(self._reranker.score, query, [chunk.text for chunk in chunks]),
                timeout=self._config.timeout_seconds,
            )
        except TimeoutError:
            elapsed = int((time.perf_counter() - started) * 1000)
            logger.warning(
                "rerank_timed_out",
                timeout_seconds=self._config.timeout_seconds,
                candidates=len(chunks),
                detail="returning retrieval order unchanged",
            )
            return RerankOutcome(
                chunks=chunks[:limit],
                rerank_ms=elapsed,
                applied=False,
                skipped_reason="timed_out",
                candidates_scored=0,
            )
        except Exception as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            logger.warning(
                "rerank_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:300],
                detail="returning retrieval order unchanged",
            )
            return RerankOutcome(
                chunks=chunks[:limit],
                rerank_ms=elapsed,
                applied=False,
                skipped_reason=f"failed:{type(exc).__name__}",
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        # A mismatched length means the model returned something unexpected.
        # Pairing scores to the wrong chunks would silently corrupt the order,
        # which is worse than not re-ranking at all.
        if len(scores) != len(chunks):
            logger.warning(
                "rerank_score_count_mismatch",
                expected=len(chunks),
                received=len(scores),
                detail="returning retrieval order unchanged",
            )
            return RerankOutcome(
                chunks=chunks[:limit],
                rerank_ms=elapsed_ms,
                applied=False,
                skipped_reason="score_count_mismatch",
            )

        scored = [
            replace(chunk, rerank_score=score, score=score)
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        scored.sort(key=lambda chunk: chunk.rerank_score or 0.0, reverse=True)

        moved = sum(
            1
            for new, old in zip(scored[:limit], chunks[:limit], strict=False)
            if new.chunk_id != old.chunk_id
        )
        logger.info(
            "reranked",
            candidates=len(chunks),
            returned=min(limit, len(scored)),
            positions_changed=moved,
            took_ms=elapsed_ms,
            model=self._reranker.model_name,
        )

        return RerankOutcome(
            chunks=scored[:limit],
            rerank_ms=elapsed_ms,
            applied=True,
            candidates_scored=len(chunks),
        )

    def warmup(self) -> None:
        """Load the model outside any timed operation."""
        if self._config.enabled:
            self._reranker.warmup()
