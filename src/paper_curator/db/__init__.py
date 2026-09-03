"""Relational persistence: SQLAlchemy models, sessions and repositories.

Owns the mapping between Python objects and Postgres tables. Both ``ingestion`` and
``api`` depend on this package; it depends on neither, so a paper record has exactly one
definition rather than one per consumer.

Populated in Phase 1.
"""

from __future__ import annotations
