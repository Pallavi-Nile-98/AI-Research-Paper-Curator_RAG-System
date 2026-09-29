"""Tests for the OpenSearch index definition and embedding providers.

The mapping is close to immutable once an index exists: a field's type cannot
change, an analyzer cannot be swapped, and ``index.knn`` cannot be enabled
afterwards. Getting one of these wrong means a full re-index, so the properties
that cannot be fixed later are asserted here.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from paper_curator.embedding import BGE_QUERY_PREFIX, FakeEmbeddingProvider
from paper_curator.search.mapping import (
    SCHEMA_VERSION,
    build_index_body,
    build_index_name,
)


@pytest.fixture
def body() -> dict[str, Any]:
    return build_index_body(embedding_dimensions=384)


@pytest.mark.unit
class TestVectorSearchSetup:
    def test_knn_is_enabled_at_creation(self, body: dict[str, Any]) -> None:
        """It cannot be turned on afterwards.

        An index created without it accepts knn_vector fields and then rejects
        every vector query, so the mistake only surfaces at search time.
        """
        assert body["settings"]["index"]["knn"] is True

    def test_embedding_dimension_matches_the_request(self, body: dict[str, Any]) -> None:
        """OpenSearch rejects any document whose vector is a different length."""
        assert body["mappings"]["properties"]["embedding"]["dimension"] == 384

    def test_uses_hnsw_with_cosine_similarity(self, body: dict[str, Any]) -> None:
        """Vectors are normalised to unit length, so cosine is the right measure."""
        method = body["mappings"]["properties"]["embedding"]["method"]
        assert method["name"] == "hnsw"
        assert method["space_type"] == "cosinesimil"

    def test_uses_the_lucene_engine(self, body: dict[str, Any]) -> None:
        """Lucene supports filtering during k-NN search.

        Post-filtering a k-NN result silently returns fewer than k documents,
        which is exactly what metadata-filtered retrieval would hit.
        """
        assert body["mappings"]["properties"]["embedding"]["method"]["engine"] == "lucene"

    def test_rejects_a_nonsensical_dimension(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            build_index_body(embedding_dimensions=0)


@pytest.mark.unit
class TestTextAnalysis:
    def test_text_is_indexed_under_two_analyzers(self, body: dict[str, Any]) -> None:
        """One for recall, one for exact terminology.

        Stemming makes "retrieving" match "retrieval", but it also collapses
        distinctions that matter in technical writing. The exact subfield keeps
        tokens as written so Phase 2 can weight the two and measure the result.
        """
        text = body["mappings"]["properties"]["text"]
        assert text["analyzer"] == "paper_english"
        assert text["fields"]["exact"]["analyzer"] == "paper_exact"

    def test_the_english_analyzer_stems_and_removes_stopwords(self, body: dict[str, Any]) -> None:
        filters = body["settings"]["analysis"]["analyzer"]["paper_english"]["filter"]
        assert "english_stemmer" in filters
        assert "english_stop" in filters

    def test_the_exact_analyzer_does_not_stem(self, body: dict[str, Any]) -> None:
        filters = body["settings"]["analysis"]["analyzer"]["paper_exact"]["filter"]
        assert "english_stemmer" not in filters


@pytest.mark.unit
class TestFieldTypes:
    @pytest.mark.parametrize(
        "field", ["arxiv_id", "categories", "primary_category", "section", "chunk_id"]
    )
    def test_filterable_fields_are_keywords(self, body: dict[str, Any], field: str) -> None:
        """A text field is tokenised, so filtering on one matches too much.

        Filtering categories = "cs.CL" against a text field would also match a
        document in "cs" or in "CL".
        """
        assert body["mappings"]["properties"][field]["type"] == "keyword"

    def test_authors_are_both_searchable_and_filterable(self, body: dict[str, Any]) -> None:
        """Keyword filters an exact name; text finds "Vaswani" in "Ashish Vaswani"."""
        authors = body["mappings"]["properties"]["authors"]
        assert authors["type"] == "text"
        assert authors["fields"]["keyword"]["type"] == "keyword"

    def test_dates_are_date_typed(self, body: dict[str, Any]) -> None:
        """Range filters need a real date type, not a string."""
        assert body["mappings"]["properties"]["published_at"]["type"] == "date"
        assert body["mappings"]["properties"]["updated_at"]["type"] == "date"

    def test_urls_are_stored_but_not_indexed(self, body: dict[str, Any]) -> None:
        """Returned with results, but never matched against a text query."""
        assert body["mappings"]["properties"]["abs_url"]["index"] is False
        assert body["mappings"]["properties"]["pdf_url"]["index"] is False

    def test_mapping_is_strict(self, body: dict[str, Any]) -> None:
        """Dynamic mapping would let one malformed document define a field type.

        Permanently, for the whole index.
        """
        assert body["mappings"]["dynamic"] == "strict"

    def test_embedding_model_is_recorded_per_document(self, body: dict[str, Any]) -> None:
        """Changing the model invalidates every stored vector."""
        assert body["mappings"]["properties"]["embedding_model"]["type"] == "keyword"


@pytest.mark.unit
class TestIndexSettings:
    def test_replicas_default_to_zero(self, body: dict[str, Any]) -> None:
        """A replica cannot sit on the node holding its primary.

        On a single node it stays unassigned and the cluster reports yellow
        forever. Zero keeps local development green; production overrides it.
        """
        assert body["settings"]["index"]["number_of_replicas"] == 0

    def test_production_can_request_replicas(self) -> None:
        body = build_index_body(embedding_dimensions=384, number_of_replicas=2)
        assert body["settings"]["index"]["number_of_replicas"] == 2


@pytest.mark.unit
class TestIndexNaming:
    def test_name_carries_the_schema_version(self) -> None:
        """Two schema versions coexist during a re-index."""
        assert f"-v{SCHEMA_VERSION}-" in build_index_name("papers")

    def test_successive_builds_get_distinct_names(self) -> None:
        earlier = build_index_name("papers", created_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
        later = build_index_name("papers", created_at=dt.datetime(2026, 6, 1, tzinfo=dt.UTC))
        assert earlier != later

    def test_two_indices_in_the_same_second_do_not_collide(self) -> None:
        """Regression: a timestamp alone is not unique enough.

        Re-indexing twice within one second produced the same name and failed
        with resource_already_exists_exception, an error that says nothing
        about the real cause.
        """
        moment = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        names = {build_index_name("papers", created_at=moment) for _ in range(50)}
        assert len(names) == 50

    def test_name_remains_chronologically_sortable(self) -> None:
        """The timestamp still leads, so _cat/indices sorts by age."""
        earlier = build_index_name("p", created_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
        later = build_index_name("p", created_at=dt.datetime(2026, 2, 1, tzinfo=dt.UTC))
        assert earlier < later

    def test_name_starts_with_the_prefix(self) -> None:
        assert build_index_name("papers").startswith("papers-")


@pytest.mark.unit
class TestFakeEmbeddingProvider:
    def test_vectors_have_the_requested_dimensionality(self) -> None:
        provider = FakeEmbeddingProvider(dimensions=384)
        assert len(provider.embed_documents(["text"])[0]) == 384

    def test_the_same_text_always_gives_the_same_vector(self) -> None:
        """Determinism is what makes indexing tests reproducible."""
        provider = FakeEmbeddingProvider()
        assert provider.embed_documents(["same"]) == provider.embed_documents(["same"])

    def test_different_texts_give_different_vectors(self) -> None:
        provider = FakeEmbeddingProvider()
        first, second = provider.embed_documents(["one", "two"])
        assert first != second

    def test_vectors_are_unit_length(self) -> None:
        """Cosine similarity reduces to a dot product only for unit vectors."""
        import math

        vector = FakeEmbeddingProvider(dimensions=64).embed_documents(["text"])[0]
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)

    def test_a_query_embeds_differently_from_the_same_text_as_a_passage(self) -> None:
        """Mirrors the asymmetry of the real model.

        A test that accidentally embeds a query through the document path then
        shows a difference, rather than passing silently.
        """
        provider = FakeEmbeddingProvider()
        assert provider.embed_query("text") != provider.embed_documents(["text"])[0]

    def test_batches_preserve_order(self) -> None:
        provider = FakeEmbeddingProvider()
        batch = provider.embed_documents(["a", "b", "c"])
        assert batch[1] == provider.embed_documents(["b"])[0]

    def test_empty_batch(self) -> None:
        assert FakeEmbeddingProvider().embed_documents([]) == []

    def test_rejects_a_nonsensical_dimensionality(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            FakeEmbeddingProvider(dimensions=0)


@pytest.mark.unit
class TestQueryPrefix:
    def test_the_bge_instruction_prefix_is_defined(self) -> None:
        """Omitting it on the query side measurably degrades retrieval.

        And it degrades silently: the maths still works and results still come
        back, just worse ones.
        """
        assert "searching relevant passages" in BGE_QUERY_PREFIX
