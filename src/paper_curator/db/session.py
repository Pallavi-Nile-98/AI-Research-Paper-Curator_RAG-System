"""Async database engine and session management.

Two drivers are used deliberately, and this is worth understanding because a
mismatch produces confusing errors:

* **asyncpg** serves the application, through
  ``postgresql+asyncpg://``. The API and the ingestion pipeline are I/O-bound —
  waiting on Postgres, OpenSearch, arXiv and the model server — so blocking the
  event loop on a database round trip would serialise requests that could
  overlap.
* **psycopg** serves Alembic, through ``postgresql+psycopg://``. Alembic's
  migration API is synchronous, and running it through an async driver requires
  a wrapper that adds nothing here. Migrations are a one-shot operation where
  concurrency is irrelevant.

Both DSNs are built by :mod:`paper_curator.core.config` from the same settings,
so they cannot drift apart.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.logging import get_logger

logger = get_logger(__name__)

# Module-level singletons. A connection pool is expensive to create and is
# designed to be shared; building one per request would defeat its purpose.
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def create_engine(settings: Settings | None = None) -> AsyncEngine:
    """Build a new async engine. Prefer :func:`get_engine` for normal use."""
    settings = settings or get_settings()
    pg = settings.postgres

    return create_async_engine(
        pg.async_dsn,
        echo=pg.echo_sql,
        pool_size=pg.pool_size,
        max_overflow=pg.max_overflow,
        # Recycle connections before common network middleboxes drop them
        # silently, which otherwise surfaces as an unexplained connection reset
        # under light load.
        pool_recycle=1800,
        # Validate a connection before handing it out. Costs one round trip and
        # removes an entire class of "server closed the connection unexpectedly"
        # errors after a database restart.
        pool_pre_ping=True,
    )


def get_engine() -> AsyncEngine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = create_engine()
        logger.debug("database_engine_created")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            # Critical for async. With the default (True), every attribute access
            # after commit triggers a lazy refresh -- which in async code raises
            # MissingGreenlet rather than quietly issuing a query. Returning an
            # ORM object from a function that just committed is a normal thing to
            # do, so this default is changed.
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Provide a transactional session: commit on success, roll back on error.

    The pipeline's unit of work. Nothing partially written survives an
    exception, which is what makes a crashed ingestion run resumable rather than
    leaving the database in a state nobody can reason about.

        async with session_scope() as session:
            session.add(paper)
    """
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session for one request.

    Separate from :func:`session_scope` because the commit boundary differs: a
    request handler decides when to commit, while the pipeline's unit of work is
    the whole block.
    """
    async with get_session_factory()() as session:
        yield session


async def dispose_engine() -> None:
    """Close all pooled connections. Call on application shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.debug("database_engine_disposed")
    _engine = None
    _session_factory = None
