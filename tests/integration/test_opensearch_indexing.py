"""Integration tests against a real OpenSearch cluster.

These exercise what unit tests structurally cannot: whether OpenSearch actually
*accepts* the mapping, whether k-NN vectors round-trip, and whether re-indexing
is genuinely idempotent rather than idempotent-by-intention.

They are marked ``integration`` and excluded from CI, which has no cluster. Run
them locally with the stack up:

    docker-compose up -d
    .venv/Scripts/python.exe -m pytest -m integration

Each test works in its own throwaway index and deletes it afterwards, so a
failed run leaves nothing behind and the application's own alias is never
touched.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from opensearchpy import AsyncOpenSearch

from paper_curator.core.config import (
    EmbeddingSettings,
    OpenSearchSettings,
    Settings,
)
from paper_curator.embedding import FakeEmbeddingProvider
from paper_curator.ingestion.chunking import TextChunk, build_chunk_document
from paper_curator.search import ChunkIndexer, IndexManager, create_client, ping
from paper_curator.search.client import OpenSearchError

pytestmark = [pytest.mark.integration]

DIMENSIONS = 384


def isolated_settings() -> Settings:
    """Build settings pointing at a unique alias, so tests cannot collide."""
    unique = uuid.uuid4().hex[:8]
    return Settings(
        opensearch=OpenSearchSettings(
            index_alias=f"test-alias-{unique}",
            index_prefix=f"test-papers-{unique}",
            bulk_batch_size=50,
        ),
        embedding=EmbeddingSettings(dimensions=DIMENSIONS),
    )


@pytest.fixture
async def settings() -> Settings:
    return isolated_settings()


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncOpenSearch]:
    opensearch = create_client(settings)
    if not await ping(opensearch):
        await opensearch.close()
        pytest.skip("OpenSearch is not reachable; run docker-compose up -d")
    yield opensearch
    await opensearch.close()


@pytest.fixture
async def manager(client: AsyncOpenSearch, settings: Settings) -> AsyncIterator[IndexManager]:
    index_manager = IndexManager(client, settings)
    yield index_manager
    # Tear down whatever the test created, alias first so deletion is allowed.
    try:
        current = await index_manager.resolve_alias()
        if current:
            await client.indices.delete_alias(index=current, name=index_manager.alias)
            await client.indices.delete(index=current)
    except Exception as exc:
        # Reported rather than swallowed. A cleanup failure leaves a stray
        # index behind, which is worth seeing even though it must not turn a
        # passing test red.
        print(f"index cleanup failed, leaving a stray index behind: {exc}")  # noqa: T201


def make_document(
    *,
    arxiv_id: str = "2401.12345",
    version: int = 1,
    chunk_index: int = 0,
    text: str = "Hybrid retrieval combines lexical scoring with dense vectors.",
    section: str | None = "Method",
) -> dict[str, Any]:
    """Build a complete chunk document, embedding included."""
    chunk = TextChunk(
        text=text,
        chunk_index=chunk_index,
        token_count=12,
        section=section,
    )
    document = build_chunk_document(
        chunk,
        arxiv_id=arxiv_id,
        version=version,
        title="A Paper About Retrieval",
        authors=["Jane Doe", "Rahul Mehta"],
        abstract="An abstract about retrieval methods.",
        published_at=dt.datetime(2024, 1, 20, tzinfo=dt.UTC),
        updated_at=dt.datetime(2024, 2, 1, tzinfo=dt.UTC),
        primary_category="cs.IR",
        categories=["cs.IR", "cs.CL"],
        abs_url=f"https://arxiv.org/abs/{arxiv_id}v{version}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}v{version}",
    )
    provider = FakeEmbeddingProvider(dimensions=DIMENSIONS)
    document["embedding"] = provider.embed_documents([text])[0]
    document["embedding_model"] = provider.model_name
    document["indexed_at"] = dt.datetime.now(dt.UTC).isoformat()
    return document


class TestIndexLifecycle:
    async def test_opensearch_accepts_the_mapping(self, manager: IndexManager) -> None:
        """The real test of the mapping: whether the cluster takes it.

        Unit tests check the dictionary's shape. Only OpenSearch can say
        whether the analyzers resolve and the knn_vector method is valid.
        """
        index = await manager.create_index()
        assert await manager.index_exists(index)

    async def test_ensure_ready_creates_and_points_the_alias(self, manager: IndexManager) -> None:
        index = await manager.ensure_ready()
        assert await manager.resolve_alias() == index

    async def test_ensure_ready_is_idempotent(self, manager: IndexManager) -> None:
        """Safe to call on every application boot."""
        first = await manager.ensure_ready()
        second = await manager.ensure_ready()
        assert first == second

    async def test_alias_switch_is_atomic(
        self, manager: IndexManager, client: AsyncOpenSearch
    ) -> None:
        """The mechanism that makes a mapping change possible without downtime."""
        original = await manager.ensure_ready()
        replacement = await manager.create_index()

        previous = await manager.switch_alias(replacement)

        assert previous == original
        assert await manager.resolve_alias() == replacement
        # The old index still exists; only the alias moved.
        assert await manager.index_exists(original)
        await client.indices.delete(index=original)

    async def test_refuses_to_delete_the_live_index(self, manager: IndexManager) -> None:
        """Deleting it would leave the alias dangling and every search failing."""
        live = await manager.ensure_ready()
        with pytest.raises(OpenSearchError, match="still points at it"):
            await manager.delete_index(live)


class TestIndexingDocuments:
    async def test_documents_are_written_and_counted(
        self, manager: IndexManager, client: AsyncOpenSearch, settings: Settings
    ) -> None:
        await manager.ensure_ready()
        indexer = ChunkIndexer(client, settings)

        documents = [make_document(chunk_index=n) for n in range(5)]
        result = await indexer.index_documents(documents)

        assert result.is_complete
        assert result.indexed_count == 5
        assert await indexer.count() == 5

    async def test_reindexing_overwrites_rather_than_duplicating(
        self, manager: IndexManager, client: AsyncOpenSearch, settings: Settings
    ) -> None:
        """The idempotency guarantee, verified against a real cluster.

        Deterministic document ids mean an interrupted run can simply be
        re-run. Without them a retry would silently double the corpus, and the
        damage would only appear as strangely repetitive search results.
        """
        await manager.ensure_ready()
        indexer = ChunkIndexer(client, settings)
        documents = [make_document(chunk_index=n) for n in range(5)]

        await indexer.index_documents(documents)
        await indexer.index_documents(documents)
        await indexer.index_documents(documents)

        assert await indexer.count() == 5

    async def test_a_document_round_trips_with_its_vector(
        self, manager: IndexManager, client: AsyncOpenSearch, settings: Settings
    ) -> None:
        """Proves the knn_vector field stores and returns a real embedding."""
        await manager.ensure_ready()
        indexer = ChunkIndexer(client, settings)
        document = make_document()
        await indexer.index_documents([document])

        stored = await client.get(index=settings.opensearch.index_alias, id=document["chunk_id"])
        source = stored["_source"]

        assert source["arxiv_id"] == "2401.12345"
        assert source["section"] == "Method"
        assert source["authors"] == ["Jane Doe", "Rahul Mehta"]
        assert len(source["embedding"]) == DIMENSIONS
        assert source["content_hash"] == document["content_hash"]

    async def test_an_empty_batch_does_nothing(
        self, client: AsyncOpenSearch, settings: Settings
    ) -> None:
        result = await ChunkIndexer(client, settings).index_documents([])
        assert result.indexed_count == 0
        assert result.is_complete

    async def test_a_document_without_a_chunk_id_is_rejected(
        self, client: AsyncOpenSearch, settings: Settings
    ) -> None:
        """Without one, OpenSearch would assign an id and break idempotency."""
        with pytest.raises(ValueError, match="chunk_id"):
            await ChunkIndexer(client, settings).index_documents([{"text": "no id"}])


class TestPartialFailure:
    async def test_good_documents_survive_a_bad_one(
        self, manager: IndexManager, client: AsyncOpenSearch, settings: Settings
    ) -> None:
        """Three bad documents must not discard four hundred good ones.

        The failed ids are reported so PostgreSQL marks only the successes,
        leaving the rest with indexed_at IS NULL for the next run to retry.
        """
        await manager.ensure_ready()
        indexer = ChunkIndexer(client, settings)

        good = [make_document(chunk_index=n) for n in range(3)]
        # A vector of the wrong length is rejected by the mapping.
        broken = make_document(chunk_index=99)
        broken["embedding"] = [0.1, 0.2, 0.3]

        result = await indexer.index_documents([*good, broken])

        assert result.indexed_count == 3
        assert result.failure_count == 1
        assert not result.is_complete
        assert broken["chunk_id"] in {failure.chunk_id for failure in result.failures}
        assert await indexer.count() == 3

    async def test_a_malformed_document_is_not_retryable(
        self, manager: IndexManager, client: AsyncOpenSearch, settings: Settings
    ) -> None:
        """Re-sending a wrong-length vector fails identically every time."""
        await manager.ensure_ready()
        indexer = ChunkIndexer(client, settings)

        broken = make_document()
        broken["embedding"] = [0.1]
        result = await indexer.index_documents([broken])

        assert result.failure_count == 1
        assert result.failures[0].is_retryable is False


class TestPaperDeletion:
    async def test_removes_only_the_named_paper(
        self, manager: IndexManager, client: AsyncOpenSearch, settings: Settings
    ) -> None:
        """Superseded versions must not stay searchable forever.

        A new version produces different chunk ids, so it never overwrites the
        old one's documents.
        """
        await manager.ensure_ready()
        indexer = ChunkIndexer(client, settings)

        await indexer.index_documents(
            [
                make_document(arxiv_id="2401.00001", chunk_index=0),
                make_document(arxiv_id="2401.00001", chunk_index=1),
                make_document(arxiv_id="2401.00002", chunk_index=0),
            ]
        )
        assert await indexer.count() == 3

        deleted = await indexer.delete_by_paper("2401.00001")

        assert deleted == 2
        assert await indexer.count() == 1

    async def test_can_target_a_single_version(
        self, manager: IndexManager, client: AsyncOpenSearch, settings: Settings
    ) -> None:
        await manager.ensure_ready()
        indexer = ChunkIndexer(client, settings)

        await indexer.index_documents(
            [
                make_document(arxiv_id="2401.00003", version=1, chunk_index=0),
                make_document(arxiv_id="2401.00003", version=2, chunk_index=0),
            ]
        )

        assert await indexer.delete_by_paper("2401.00003", version=1) == 1
        assert await indexer.count() == 1
