"""The OpenSearch index definition.

This mapping is where ADR-0004 becomes concrete: **one document holds both the
chunk's text and its vector**, so a lexical query and a semantic query score the
same records and their results can be fused without a join across systems.

Four decisions here are worth understanding, because each one silently degrades
retrieval if taken the other way.

**k-NN must be enabled at index creation.** ``index.knn`` cannot be turned on
afterwards. An index created without it accepts ``knn_vector`` fields and then
rejects every vector query, so getting it wrong means a full re-index.

**Text is indexed twice, under different analyzers.** ``text`` uses the English
analyzer, which lowercases, strips stopwords and stems -- so a search for
"retrieving" matches "retrieval". ``text.exact`` keeps tokens as written. That
second field is what makes exact terminology searchable: stemming maps
"BM25" and "BM-25" apart while collapsing distinctions that matter in technical
writing. Phase 2 can weight the two against each other and measure the result.

**Filterable fields are keywords, searchable fields are text.** A ``text`` field
is tokenised, so filtering ``categories = "cs.CL"`` against one would also match
a document in "cs" or "CL". Author names get both: ``keyword`` for exact
filtering, ``text`` for searching "Vaswani" inside "Ashish Vaswani".

**Replicas default to zero.** A replica shard cannot be placed on the node that
holds its primary, so on a single-node cluster it stays unassigned forever and
the cluster reports yellow. Zero replicas keeps local development green and is
explicitly wrong for production, where the deployment overrides it.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Final

# Bumped when the mapping changes in a way that requires re-indexing: a new
# analyzer, a changed vector dimension, a field whose type changed. Part of the
# physical index name, so two schema versions can coexist during a migration.
SCHEMA_VERSION: Final = 1

# HNSW graph construction. Higher ef_construction builds a more accurate graph
# more slowly; higher m gives each node more edges, improving recall at the cost
# of memory. These are the usual starting values, and Phase 2 measures whether
# they are worth changing.
HNSW_EF_CONSTRUCTION: Final = 128
HNSW_M: Final = 16


def build_index_name(prefix: str, *, created_at: dt.datetime | None = None) -> str:
    """Return a versioned physical index name.

    Reads and writes always go through an alias, never this name. Re-indexing
    therefore means building a new physical index alongside the live one and
    swapping the alias atomically, rather than taking search down while an index
    is rebuilt.

    The name carries a timestamp so successive rebuilds sort chronologically in
    a ``_cat/indices`` listing, and a short random suffix so two indices created
    in the same second cannot collide. An earlier version used the timestamp
    alone and failed with ``resource_already_exists_exception`` whenever a
    re-index ran twice inside one second -- an error that says nothing about the
    actual cause.
    """
    stamp = (created_at or dt.datetime.now(dt.UTC)).strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{prefix}-v{SCHEMA_VERSION}-{stamp}-{suffix}"


def build_index_body(
    *,
    embedding_dimensions: int,
    number_of_shards: int = 1,
    number_of_replicas: int = 0,
) -> dict[str, Any]:
    """Build the settings and mappings for a new chunk index.

    Args:
        embedding_dimensions: Vector length. **Must** equal the embedding
            model's output dimension -- OpenSearch rejects any document whose
            vector is a different length, and the mapping cannot be changed
            afterwards.
        number_of_shards: One is correct well past this project's scale. A
            shard is the unit of parallelism, and splitting a small corpus
            across several only adds coordination overhead.
        number_of_replicas: Zero for single-node development. Production sets
            at least one, for redundancy and read throughput.

    """
    if embedding_dimensions < 1:
        msg = f"embedding_dimensions must be positive, got {embedding_dimensions}"
        raise ValueError(msg)

    return {
        "settings": {
            "index": {
                # Must be set at creation; it cannot be enabled later.
                "knn": True,
                "number_of_shards": number_of_shards,
                "number_of_replicas": number_of_replicas,
                # How soon a written document becomes searchable. One second is
                # the default and is right for a system that ingests
                # continuously; bulk backfills can raise it temporarily.
                "refresh_interval": "1s",
            },
            "analysis": {
                "analyzer": {
                    # Recall-oriented: stems, lowercases, strips stopwords.
                    "paper_english": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "english_stop", "english_stemmer"],
                    },
                    # Precision-oriented: no stemming, so exact terminology and
                    # identifiers survive intact.
                    "paper_exact": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase"],
                    },
                },
                "filter": {
                    "english_stop": {"type": "stop", "stopwords": "_english_"},
                    "english_stemmer": {"type": "stemmer", "language": "english"},
                },
            },
        },
        "mappings": {
            # Reject documents with unmapped fields rather than guessing a type
            # for them. Dynamic mapping would let one malformed document define
            # a field's type for the whole index, permanently.
            "dynamic": "strict",
            "properties": {
                # --- Identity -------------------------------------------------
                # Deterministic, so re-indexing overwrites instead of duplicating.
                "chunk_id": {"type": "keyword"},
                "content_hash": {"type": "keyword"},
                # --- Retrievable content --------------------------------------
                "text": {
                    "type": "text",
                    "analyzer": "paper_english",
                    "fields": {
                        "exact": {"type": "text", "analyzer": "paper_exact"},
                    },
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dimensions,
                    "method": {
                        "name": "hnsw",
                        # Vectors are normalised to unit length, so cosine
                        # similarity is the right measure.
                        "space_type": "cosinesimil",
                        # The Lucene engine supports efficient filtering during
                        # k-NN search, which metadata-filtered retrieval needs.
                        # Post-filtering a k-NN result silently returns fewer
                        # than k documents.
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": HNSW_EF_CONSTRUCTION,
                            "m": HNSW_M,
                        },
                    },
                },
                # Recorded per document: changing the model invalidates every
                # stored vector, and this is how a stale one is detected.
                "embedding_model": {"type": "keyword"},
                # --- Position within the paper --------------------------------
                "chunk_index": {"type": "integer"},
                "section": {"type": "keyword"},
                "subsection": {"type": "keyword"},
                "section_kind": {"type": "keyword"},
                "token_count": {"type": "integer"},
                "char_count": {"type": "integer"},
                # --- Paper identity, needed to render a citation ---------------
                "arxiv_id": {"type": "keyword"},
                "version": {"type": "integer"},
                "title": {
                    "type": "text",
                    "analyzer": "paper_english",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
                },
                "authors": {
                    # Both, deliberately: keyword filters on an exact name,
                    # text finds "Vaswani" inside "Ashish Vaswani".
                    "type": "text",
                    "analyzer": "paper_exact",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "abstract": {"type": "text", "analyzer": "paper_english"},
                # --- Filterable metadata ---------------------------------------
                "published_at": {"type": "date"},
                "updated_at": {"type": "date"},
                "primary_category": {"type": "keyword"},
                "categories": {"type": "keyword"},
                # --- Links shown to the reader ---------------------------------
                # index=false: stored and returned, but never searched. Saves
                # index space and stops a URL matching a text query.
                "abs_url": {"type": "keyword", "index": False},
                "pdf_url": {"type": "keyword", "index": False},
                # --- Bookkeeping ------------------------------------------------
                "indexed_at": {"type": "date"},
            },
        },
    }
