"""Embeddings from a local sentence-transformers model.

Imports of ``sentence_transformers`` are deliberately deferred to first use.
The package pulls in PyTorch -- roughly 2 GB -- and lives behind the optional
``[ml]`` extra, so importing it at module level would make every part of the
codebase that merely mentions embeddings unusable without it, including CI,
which needs no model to run unit tests.

The default model is ``BAAI/bge-small-en-v1.5``: 384 dimensions, around 130 MB,
and fast enough on a CPU to embed a paper's worth of chunks in seconds. Two of
its properties shape this module.

**It is asymmetric.** Queries are trained with an instruction prefix and
passages without one. Embedding a query as though it were a passage measurably
degrades retrieval -- and degrades it *silently*, since the maths still works
and results still come back.

**It normalises to unit length.** With unit vectors, cosine similarity reduces
to a dot product, which is what OpenSearch's k-NN scoring computes. Skipping
normalisation makes long passages score higher simply for being long.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.exceptions import ConfigurationError
from paper_curator.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    pass

logger = get_logger(__name__)

# Instruction prefix the BGE family expects on the query side. Applied only to
# queries; adding it to passages hurts retrieval just as omitting it does.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Model-name prefixes whose query side needs the instruction above.
_ASYMMETRIC_PREFIXES = ("baai/bge-", "bge-")


def _needs_query_prefix(model_name: str) -> bool:
    """Whether this model expects a query-side instruction prefix."""
    lowered = model_name.lower()
    # The v1.5 "-en-" English models use it; multilingual e5-style models have
    # their own conventions and are not assumed here.
    return any(lowered.startswith(prefix) for prefix in _ASYMMETRIC_PREFIXES)


class SentenceTransformerEmbeddingProvider:
    """Embeds text with a locally loaded sentence-transformers model.

    The model is loaded lazily on first use rather than in ``__init__``, so
    constructing the provider is cheap and a process that never embeds anything
    never pays the load cost.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        config = (settings or get_settings()).embedding
        self._model_name = config.model_name
        self._configured_dimensions = config.dimensions
        self._batch_size = config.batch_size
        self._device = config.device
        self._normalize = config.normalize
        self._model: Any | None = None
        self._uses_query_prefix = _needs_query_prefix(self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Vector length, verified against the loaded model where possible.

        The configured value must match the ``knn_vector`` dimension in the
        OpenSearch mapping. Once the model is loaded its real dimensionality is
        authoritative, and a mismatch with configuration is reported rather
        than silently tolerated -- OpenSearch would otherwise reject every
        document with an unhelpful error.
        """
        if self._model is None:
            return self._configured_dimensions
        actual = int(self._model.get_sentence_embedding_dimension())
        if actual != self._configured_dimensions:
            msg = (
                f"model {self._model_name!r} produces {actual}-dimensional vectors "
                f"but EMBEDDING_DIMENSIONS is {self._configured_dimensions}; "
                "the OpenSearch mapping must match the model"
            )
            raise ConfigurationError(msg)
        return actual

    def _load(self) -> Any:
        """Load the model on first use."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            msg = (
                "sentence-transformers is not installed. Install the optional "
                'extra with: pip install -e ".[ml]"'
            )
            raise ConfigurationError(msg) from exc

        logger.info("embedding_model_loading", model=self._model_name, device=self._device)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        # Surfaces a dimension mismatch at load time rather than at the first
        # rejected bulk write.
        _ = self.dimensions
        logger.info(
            "embedding_model_loaded",
            model=self._model_name,
            dimensions=self._configured_dimensions,
            asymmetric=self._uses_query_prefix,
        )
        return self._model

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        if not texts:
            return []
        vectors = self._encode(texts)
        logger.debug("documents_embedded", count=len(vectors), model=self._model_name)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query, applying the instruction prefix if required."""
        prepared = f"{BGE_QUERY_PREFIX}{text}" if self._uses_query_prefix else text
        return self._encode([prepared])[0]
