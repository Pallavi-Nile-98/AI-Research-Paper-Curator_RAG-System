"""Text embedding for vector retrieval.

    from paper_curator.embedding import get_embedding_provider

    provider = get_embedding_provider()
    vectors = provider.embed_documents(["a passage", "another passage"])
    query = provider.embed_query("what is BM25?")

A top-level package rather than a part of ``ingestion`` because both ends of the
system use it: ingestion embeds chunks at index time, retrieval embeds the
question at query time. They must use the same model -- vectors from different
models are not comparable, and comparing them produces plausible nonsense rather
than an error.
"""

from paper_curator.core.config import Settings, get_settings
from paper_curator.embedding.base import EmbeddingProvider, FakeEmbeddingProvider
from paper_curator.embedding.sentence_transformers import (
    BGE_QUERY_PREFIX,
    SentenceTransformerEmbeddingProvider,
)

__all__ = [
    "BGE_QUERY_PREFIX",
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "get_embedding_provider",
]


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Build the configured embedding provider.

    The indirection exists so the provider can be swapped by configuration --
    for a different model, or for the fake in tests -- without any caller
    knowing which implementation it holds.
    """
    resolved = settings or get_settings()
    # One provider today, so no branch. EMBEDDING_PROVIDER is a Literal, which
    # means adding a second one is a type error here until this function
    # handles it -- the check happens at type-check time rather than as a
    # runtime ValueError nobody would hit.
    return SentenceTransformerEmbeddingProvider(resolved)
