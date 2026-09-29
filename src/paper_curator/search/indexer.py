"""Bulk writes into OpenSearch.

Two properties matter more than throughput here.

**Writes are idempotent.** Every document is written with a deterministic
``_id`` -- the chunk's ``chunk_id`` -- using the ``index`` action, which
replaces any existing document with that id. Re-running an interrupted
ingestion therefore overwrites rather than duplicating. Letting OpenSearch
generate ids would make a re-run silently double the corpus, and the damage
would only become visible as strangely repetitive search results.

**Partial failure is the normal case, not an exception.** A bulk request of 500
documents can have 3 rejected and 497 accepted. Treating that as a failed batch
loses 497 good documents; treating it as a success loses track of 3 bad ones.
So the result reports exactly which ids succeeded, and the caller marks only
those as indexed in PostgreSQL. The rest keep ``indexed_at IS NULL`` and are
picked up by the next run -- which is the reconciliation mechanism ADR-0002
describes, made concrete.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import OpenSearchException
from opensearchpy.helpers import async_bulk

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.logging import get_logger
from paper_curator.search.client import OpenSearchError

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IndexFailure:
    """One document OpenSearch refused, and why."""

    chunk_id: str
    reason: str
    status: int | None = None

    @property
    def is_retryable(self) -> bool:
        """Whether re-sending this document could succeed.

        A 429 means the cluster shed load under pressure and the document is
        fine. A 400 means the document is wrong -- most often a vector whose
        length does not match the mapping -- and re-sending it will fail
        identically.
        """
        if self.status is None:
            return True
        return self.status == 429 or self.status >= 500


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Outcome of one indexing operation, split by document."""

    indexed_ids: list[str] = field(default_factory=list)
    failures: list[IndexFailure] = field(default_factory=list)

    @property
    def indexed_count(self) -> int:
        return len(self.indexed_ids)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def is_complete(self) -> bool:
        """True when every document was accepted."""
        return not self.failures

    @property
    def retryable_failures(self) -> list[IndexFailure]:
        return [failure for failure in self.failures if failure.is_retryable]

    def summary(self) -> str:
        """One-line description for logs."""
        if self.is_complete:
            return f"{self.indexed_count} indexed"
        return (
            f"{self.indexed_count} indexed, {self.failure_count} failed "
            f"({len(self.retryable_failures)} retryable)"
        )


class ChunkIndexer:
    """Writes chunk documents into the index behind the alias."""

    def __init__(self, client: AsyncOpenSearch, settings: Settings | None = None) -> None:
        config = settings or get_settings()
        self._client = client
        self._alias = config.opensearch.index_alias
        self._batch_size = config.opensearch.bulk_batch_size

    def _actions(self, documents: Sequence[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        """Build bulk actions with deterministic document ids."""
        for document in documents:
            chunk_id = document.get("chunk_id")
            if not chunk_id:
                msg = "every chunk document must carry a chunk_id for idempotent indexing"
                raise ValueError(msg)
            yield {
                # "index" replaces an existing document with this id. "create"
                # would fail on a re-run, and letting OpenSearch assign an id
                # would duplicate the corpus.
                "_op_type": "index",
                "_index": self._alias,
                "_id": chunk_id,
                "_source": document,
            }

    async def index_documents(self, documents: Sequence[dict[str, Any]]) -> IndexResult:
        """Write documents in batches, reporting each one's outcome.

        Raises:
            OpenSearchError: The request could not be made at all -- the cluster
                is unreachable, or the alias is unusable. A rejected *document*
                is reported in the result instead, not raised.

        """
        if not documents:
            return IndexResult()

        try:
            succeeded, raw_errors = await async_bulk(
                self._client,
                self._actions(documents),
                chunk_size=self._batch_size,
                # Report per-document failures rather than aborting the batch,
                # so 3 bad documents cannot discard 497 good ones.
                raise_on_error=False,
                raise_on_exception=False,
                # Make the writes searchable before returning, so a caller can
                # verify what it just wrote without waiting for the next
                # refresh interval.
                refresh=True,
            )
        except OpenSearchException as exc:
            msg = f"bulk indexing request failed: {exc}"
            raise OpenSearchError(msg, context={"documents": len(documents)}) from exc

        # async_bulk returns either a list of failed items or, with
        # stats_only, a bare count. We never request stats_only, but the
        # signature allows both and a silently-ignored int would mean treating
        # failed documents as indexed.
        failures = _parse_failures(raw_errors if isinstance(raw_errors, list) else [])
        failed_ids = {failure.chunk_id for failure in failures}
        indexed_ids = [
            str(document["chunk_id"])
            for document in documents
            if document["chunk_id"] not in failed_ids
        ]

        result = IndexResult(indexed_ids=indexed_ids, failures=failures)

        # `succeeded` is the library's own count; a mismatch with our own means
        # a response shape we did not anticipate, and silently trusting either
        # number would corrupt what PostgreSQL believes is indexed.
        if succeeded != result.indexed_count:
            logger.warning(
                "bulk_count_mismatch",
                library_reported=succeeded,
                derived=result.indexed_count,
                detail="treating the derived per-document outcome as authoritative",
            )

        log = logger.info if result.is_complete else logger.warning
        log(
            "chunks_indexed",
            alias=self._alias,
            indexed=result.indexed_count,
            failed=result.failure_count,
            retryable=len(result.retryable_failures),
        )
        for failure in result.failures[:5]:
            logger.warning(
                "chunk_index_failed",
                chunk_id=failure.chunk_id,
                status=failure.status,
                reason=failure.reason,
                retryable=failure.is_retryable,
            )
        return result

    async def delete_by_paper(self, arxiv_id: str, version: int | None = None) -> int:
        """Remove a paper's chunks from the index.

        Used when a newer version supersedes an older one: the old version's
        chunks would otherwise remain searchable forever, since a new version
        produces different chunk ids and so never overwrites them.

        Returns the number of documents deleted.
        """
        must: list[dict[str, Any]] = [{"term": {"arxiv_id": arxiv_id}}]
        if version is not None:
            must.append({"term": {"version": version}})

        try:
            response = await self._client.delete_by_query(
                index=self._alias,
                body={"query": {"bool": {"must": must}}},
                refresh=True,
                # Do not abort the whole deletion because one document was
                # updated concurrently; those are retried instead.
                conflicts="proceed",
            )
        except OpenSearchException as exc:
            msg = f"could not delete chunks for {arxiv_id}: {exc}"
            raise OpenSearchError(msg, context={"arxiv_id": arxiv_id}) from exc

        deleted = int(response.get("deleted", 0))
        logger.info("paper_chunks_deleted", arxiv_id=arxiv_id, version=version, deleted=deleted)
        return deleted

    async def count(self) -> int:
        """Return the number of documents currently visible through the alias."""
        try:
            response = await self._client.count(index=self._alias)
        except OpenSearchException as exc:
            msg = f"could not count documents in {self._alias!r}: {exc}"
            raise OpenSearchError(msg) from exc
        return int(response.get("count", 0))


def _parse_failures(raw_errors: Iterable[Any]) -> list[IndexFailure]:
    """Extract per-document failures from a bulk response.

    The shape varies: an entry may be a dict keyed by operation type, or a bare
    string when the client could not parse the response. Both are handled, since
    losing a failure here would mark a missing document as successfully indexed.
    """
    failures: list[IndexFailure] = []

    for entry in raw_errors:
        if not isinstance(entry, dict):
            failures.append(IndexFailure(chunk_id="<unknown>", reason=str(entry)))
            continue

        for operation in entry.values():
            if not isinstance(operation, dict):
                continue
            error = operation.get("error") or {}
            reason = (
                error.get("reason")
                if isinstance(error, dict)
                else str(error) or "unspecified error"
            )
            failures.append(
                IndexFailure(
                    chunk_id=str(operation.get("_id", "<unknown>")),
                    reason=str(reason or "unspecified error"),
                    status=operation.get("status"),
                )
            )

    return failures
