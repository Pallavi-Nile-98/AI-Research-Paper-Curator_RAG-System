"""Index lifecycle: creation, alias management and re-indexing.

Nothing in this system reads or writes a physical index by name. Everything goes
through an **alias**, and that indirection is what makes the mapping changeable.

An OpenSearch mapping is largely immutable. A field's type cannot be changed, an
analyzer cannot be swapped, and ``index.knn`` cannot be enabled after creation.
Changing any of them means building a new index. Without an alias that is an
outage: delete, recreate, re-ingest, and search is broken throughout. With one
it is a swap --

    1. create ``papers-v2-20260928`` alongside the live ``papers-v1-...``
    2. populate it from PostgreSQL, which is the system of record (ADR-0002)
    3. atomically point ``papers-current`` at the new index
    4. delete the old index once nothing is reading it

Step 3 is a single OpenSearch action that adds and removes aliases together, so
there is no moment when the alias points at nothing or at both.

This is only safe because OpenSearch holds nothing that cannot be rebuilt: every
indexed document is derived from rows PostgreSQL already has.
"""

from __future__ import annotations

from typing import Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError, OpenSearchException

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.logging import get_logger
from paper_curator.search.client import OpenSearchError
from paper_curator.search.mapping import build_index_body, build_index_name

logger = get_logger(__name__)


class IndexManager:
    """Creates indices and moves the read/write alias between them."""

    def __init__(self, client: AsyncOpenSearch, settings: Settings | None = None) -> None:
        self._client = client
        config = settings or get_settings()
        self._alias = config.opensearch.index_alias
        self._prefix = config.opensearch.index_prefix
        self._dimensions = config.embedding.dimensions

    @property
    def alias(self) -> str:
        """The name everything reads and writes through."""
        return self._alias

    async def create_index(self, name: str | None = None) -> str:
        """Create a new physical index and return its name.

        Does not touch the alias. A freshly created index is invisible to the
        application until :meth:`switch_alias` points at it, which is what makes
        populating it safe while the old one keeps serving searches.
        """
        index_name = name or build_index_name(self._prefix)
        body = build_index_body(embedding_dimensions=self._dimensions)

        try:
            await self._client.indices.create(index=index_name, body=body)
        except OpenSearchException as exc:
            msg = f"could not create index {index_name!r}: {exc}"
            raise OpenSearchError(msg, context={"index": index_name}) from exc

        logger.info(
            "index_created",
            index=index_name,
            dimensions=self._dimensions,
        )
        return index_name

    async def index_exists(self, name: str) -> bool:
        """Whether a physical index exists."""
        try:
            return bool(await self._client.indices.exists(index=name))
        except OpenSearchException as exc:
            msg = f"could not check index {name!r}: {exc}"
            raise OpenSearchError(msg) from exc

    async def resolve_alias(self) -> str | None:
        """Return the index the alias points at, or None when it is unset."""
        try:
            response: dict[str, Any] = await self._client.indices.get_alias(name=self._alias)
        except NotFoundError:
            return None
        except OpenSearchException as exc:
            msg = f"could not resolve alias {self._alias!r}: {exc}"
            raise OpenSearchError(msg) from exc

        names = sorted(response)
        if len(names) > 1:
            # An alias over several indices makes writes ambiguous: OpenSearch
            # refuses to index through it. Worth failing loudly rather than
            # letting the next write produce a confusing error.
            msg = (
                f"alias {self._alias!r} points at {len(names)} indices ({', '.join(names)}); "
                "writes through a multi-index alias are rejected"
            )
            raise OpenSearchError(msg, context={"indices": names})
        return names[0] if names else None

    async def switch_alias(self, to_index: str) -> str | None:
        """Point the alias at ``to_index``, atomically.

        Add and remove travel in one actions list, so the alias never points at
        nothing and never points at two indices at once -- the state in which
        writes are rejected.

        Returns the index the alias previously pointed at, or None.
        """
        previous = await self.resolve_alias()

        actions: list[dict[str, Any]] = []
        if previous is not None:
            actions.append({"remove": {"index": previous, "alias": self._alias}})
        actions.append({"add": {"index": to_index, "alias": self._alias}})

        try:
            await self._client.indices.update_aliases(body={"actions": actions})
        except OpenSearchException as exc:
            msg = f"could not switch alias {self._alias!r} to {to_index!r}: {exc}"
            raise OpenSearchError(msg, context={"index": to_index}) from exc

        logger.info("alias_switched", alias=self._alias, to=to_index, previous=previous)
        return previous

    async def ensure_ready(self) -> str:
        """Guarantee the alias resolves to a usable index, creating one if not.

        The idempotent startup path: safe to call on every boot. It creates an
        index only when the alias resolves to nothing, so restarting the
        application never disturbs existing data.
        """
        existing = await self.resolve_alias()
        if existing is not None:
            logger.debug("index_ready", alias=self._alias, index=existing)
            return existing

        index_name = await self.create_index()
        await self.switch_alias(index_name)
        logger.info("index_bootstrapped", alias=self._alias, index=index_name)
        return index_name

    async def delete_index(self, name: str) -> None:
        """Delete a physical index.

        Refuses to delete the index the alias currently points at. Doing so
        would leave the alias dangling and every search failing, and the mistake
        is easy to make while cleaning up after a re-index.
        """
        current = await self.resolve_alias()
        if current == name:
            msg = (
                f"refusing to delete {name!r}: alias {self._alias!r} still points at it. "
                "Switch the alias to another index first."
            )
            raise OpenSearchError(msg, context={"index": name, "alias": self._alias})

        try:
            await self._client.indices.delete(index=name)
        except NotFoundError:
            logger.debug("index_already_absent", index=name)
            return
        except OpenSearchException as exc:
            msg = f"could not delete index {name!r}: {exc}"
            raise OpenSearchError(msg) from exc

        logger.info("index_deleted", index=name)

    async def refresh(self) -> None:
        """Make recently written documents searchable immediately.

        OpenSearch is near-real-time: a write becomes visible at the next
        refresh, one second by default. Tests and the end of a bulk ingestion
        run need to read what was just written, and waiting a second in a loop
        is both slower and flakier than asking.
        """
        try:
            await self._client.indices.refresh(index=self._alias)
        except NotFoundError:
            return
        except OpenSearchException as exc:
            msg = f"could not refresh {self._alias!r}: {exc}"
            raise OpenSearchError(msg) from exc
