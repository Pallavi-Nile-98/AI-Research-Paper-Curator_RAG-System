"""OpenSearch connection management.

A thin wrapper over ``opensearch-py``'s async client. It exists so the rest of
the codebase depends on this module rather than on the library directly: the
connection is built once from settings, errors are translated into the
application's exception hierarchy, and tests can inject a client without
patching a third-party import.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import OpenSearchException

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.exceptions import ExternalServiceError
from paper_curator.core.logging import get_logger

logger = get_logger(__name__)

SERVICE_NAME = "opensearch"


class OpenSearchError(ExternalServiceError):
    """An OpenSearch request failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(context or {})
        if status_code is not None:
            merged["status_code"] = status_code
        super().__init__(message, service=SERVICE_NAME, context=merged)
        self.status_code = status_code

    @property
    def is_retryable(self) -> bool:
        """True when a later attempt could plausibly succeed.

        A 400 means the request itself is wrong -- a malformed query, a vector
        of the wrong length -- and will be wrong next time too. A 429 means the
        cluster is shedding load and is exactly what backoff is for.
        """
        if self.status_code is None:
            return True
        return self.status_code == 429 or self.status_code >= 500


def create_client(settings: Settings | None = None) -> AsyncOpenSearch:
    """Build an async OpenSearch client from configuration."""
    config = (settings or get_settings()).opensearch

    auth = None
    if config.username and config.password:
        auth = (config.username, config.password.get_secret_value())

    return AsyncOpenSearch(
        hosts=[{"host": config.host, "port": config.port}],
        http_auth=auth,
        use_ssl=config.use_ssl,
        # Local development runs the security plugin disabled, so there is no
        # certificate to verify. Production uses Amazon OpenSearch Service with
        # TLS, where this is true and certificates are checked.
        verify_certs=config.verify_certs,
        ssl_show_warn=config.verify_certs,
        timeout=config.request_timeout_seconds,
        # The library retries idempotent requests itself; application-level
        # retry sits above it for the operations where partial success matters.
        max_retries=config.max_retries,
        retry_on_timeout=True,
    )


async def ping(client: AsyncOpenSearch) -> bool:
    """Report whether the cluster is reachable.

    Used by the readiness probe. Returns False rather than raising, because a
    health check that throws is harder to use than one that answers.
    """
    try:
        return bool(await client.ping())
    except OpenSearchException as exc:
        logger.warning("opensearch_ping_failed", error=str(exc))
        return False


async def cluster_health(client: AsyncOpenSearch) -> dict[str, Any]:
    """Return the cluster health document.

    A single-node cluster reports ``yellow`` whenever an index requests
    replicas, because a replica cannot share a node with its primary. That is
    expected locally and is not a fault, which is why callers should treat
    yellow as healthy rather than degraded.
    """
    try:
        result: dict[str, Any] = await client.cluster.health()
    except OpenSearchException as exc:
        msg = f"could not read cluster health: {exc}"
        raise OpenSearchError(msg) from exc
    return result
