"""Alembic migration environment.

Differs from the generated template in three ways, each deliberate:

1. **The database URL comes from application settings**, not from
   ``alembic.ini``. One source of truth, and no credential in a tracked file.
2. **Logging is the application's structlog setup**, not Alembic's
   ``fileConfig``. Migration output then matches every other log this system
   emits, including in CI.
3. **Every model package is imported explicitly.** Autogenerate compares the
   database against ``Base.metadata``, and a model in a module nobody imported
   is simply absent from it — producing a migration that silently omits the
   table.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool

from paper_curator.core.config import get_settings
from paper_curator.core.logging import configure_logging
from paper_curator.db.base import Base

# Import for the side effect of registering every model on Base.metadata.
# Without this, autogenerate produces an empty migration.
import paper_curator.db.models  # noqa: F401  # isort: skip

config = context.config

settings = get_settings()
configure_logging(level=settings.app.log_level, log_format=settings.app.log_format)

target_metadata = Base.metadata


def get_database_url() -> str:
    """Return the synchronous DSN used for migrations.

    Synchronous psycopg rather than asyncpg: Alembic's migration API is
    synchronous, and concurrency is irrelevant to a one-shot schema change.
    """
    return get_settings().postgres.sync_dsn


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it.

    Used with ``alembic upgrade head --sql`` to produce a script for review, or
    for handing to a DBA in an environment where the application has no
    migration privileges.
    """
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and apply migrations."""
    # NullPool: a migration opens one connection, uses it, and exits. A pool
    # would keep connections alive after the work is done.
    engine = create_engine(get_database_url(), poolclass=pool.NullPool)

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Detect column type and server-default changes. Both are off by
            # default, which means autogenerate would silently miss a
            # VARCHAR(64) widened to VARCHAR(128).
            compare_type=True,
            compare_server_default=True,
        )

        # Wrap each migration in a transaction. PostgreSQL supports
        # transactional DDL, so a migration that fails halfway rolls back
        # completely rather than leaving a half-applied schema.
        with context.begin_transaction():
            context.run_migrations()

    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
