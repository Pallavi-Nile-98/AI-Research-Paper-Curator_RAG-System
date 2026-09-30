"""The embedding provider interface.

Embeddings sit at a boundary used from both ends of the system: ingestion embeds
chunks at index time, retrieval embeds the user's question at query time. Both
go through this protocol, which matters for one reason above all --

**the query and the documents must be embedded by the same model.** Vectors from
different models are not comparable. Their dimensions may coincide, the maths
still runs, cosine similarity still returns a number, and the results are
nonsense. Nothing errors. Routing both paths through one interface makes that
mismatch hard to introduce by accident.

Two other decisions are encoded here.

**Documents and queries are embedded differently.** Several modern embedding
models, including the default, are trained asymmetrically: a query gets an
instruction prefix and a passage does not. Using the document path for a query
measurably degrades retrieval, so the interface separates them rather than
leaving it to each caller to remember.

**The interface is synchronous.** Embedding is CPU-bound, not I/O-bound, so
``async`` would buy nothing and mislead. Callers in async contexts run it on a
worker thread, the same way PDF parsing and OCR are handled.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors for similarity search."""

    @property
    def model_name(self) -> str:
        """Identifier of the underlying model.

        Stored alongside every indexed chunk. Changing the model invalidates
        every stored vector, so knowing which model produced one is what makes
        a re-index detectable rather than a silent corruption.
        """
        ...

    @property
    def dimensions(self) -> int:
        """Length of the vectors produced.

        Must match the ``knn_vector`` dimension in the OpenSearch mapping;
        OpenSearch rejects a document whose vector is the wrong length.
        """
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for indexing, in batches."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Separate from :meth:`embed_documents` because asymmetric models expect
        a query-side instruction prefix.
        """
        ...

    def warmup(self) -> None:
        """Do any expensive one-off setup now rather than on first use.

        Part of the protocol so callers need not know whether a provider loads
        a model. Benchmarks must call it: loading takes tens of seconds while
        embedding takes milliseconds, so an unwarmed first query reports a
        latency that is almost entirely model loading.
        """
        ...


class FakeEmbeddingProvider:
    """Deterministic embeddings derived from a hash, for tests.

    Produces stable, unit-length vectors with no model, no download and no
    PyTorch, so indexing and retrieval logic can be tested in milliseconds.

    Similarity between two fake vectors is meaningless -- these test plumbing,
    not relevance. Anything measuring retrieval *quality* must use a real model,
    and the evaluation suite does.
    """

    def __init__(self, dimensions: int = 384, model_name: str = "fake-embedding-model") -> None:
        if dimensions < 1:
            msg = f"dimensions must be at least 1, got {dimensions}"
            raise ValueError(msg)
        self._dimensions = dimensions
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _vector(self, text: str, *, prefix: str = "") -> list[float]:
        """Derive a unit-length vector deterministically from the text."""
        digest = hashlib.sha256(f"{prefix}{text}".encode()).digest()
        # Repeat the digest until it covers the requested dimensionality.
        raw = (digest * (self._dimensions // len(digest) + 1))[: self._dimensions]
        # Centre on zero so vectors are not all crowded into one orthant.
        values = [(byte - 127.5) / 127.5 for byte in raw]

        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:  # pragma: no cover - only for an all-zero digest
            return [0.0] * self._dimensions
        return [v / norm for v in values]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        # Prefixed so a query and an identical passage produce different
        # vectors, mirroring the asymmetry of the real model. A test that
        # accidentally embeds a query as a document then shows a difference.
        return self._vector(text, prefix="query: ")

    def warmup(self) -> None:
        """No-op: there is nothing to load."""
