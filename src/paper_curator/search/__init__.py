"""The OpenSearch boundary: connection, index schema, lifecycle and writes.

    from paper_curator.search import ChunkIndexer, IndexManager, create_client

    client = create_client()
    await IndexManager(client).ensure_ready()
    result = await ChunkIndexer(client).index_documents(documents)

A package of its own rather than a part of ``ingestion`` or ``retrieval``,
because both use it and they must agree. Ingestion writes documents and
retrieval queries them, so the mapping is a contract between the two: an
analyzer chosen at write time determines what a query can match at read time.
Putting the schema inside either one would make the other's dependency on it
invisible.
"""

from paper_curator.search.client import (
    OpenSearchError,
    cluster_health,
    create_client,
    ping,
)
from paper_curator.search.index_manager import IndexManager
from paper_curator.search.indexer import ChunkIndexer, IndexFailure, IndexResult
from paper_curator.search.mapping import (
    SCHEMA_VERSION,
    build_index_body,
    build_index_name,
)

__all__ = [
    "SCHEMA_VERSION",
    "ChunkIndexer",
    "IndexFailure",
    "IndexManager",
    "IndexResult",
    "OpenSearchError",
    "build_index_body",
    "build_index_name",
    "cluster_health",
    "create_client",
    "ping",
]
