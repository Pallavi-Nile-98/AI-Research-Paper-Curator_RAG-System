"""Data access for the ingestion pipeline.

Plain async functions taking a session rather than repository classes. There is
no state to hold between calls -- the session *is* the state -- and a class
whose every method takes the same first argument is a namespace wearing a
costume.

Keeping the session a parameter also puts the transaction boundary where it
belongs: with the caller, who knows what constitutes one unit of work. A
repository that committed internally would make "store the paper and its
chunks, or neither" impossible to express.
"""

from paper_curator.db.repositories.papers import (
    UpsertOutcome,
    UpsertResult,
    count_unindexed_chunks,
    get_or_create_authors,
    mark_chunks_indexed,
    normalize_author_name,
    replace_chunks,
    save_document_text,
    upsert_paper,
)
from paper_curator.db.repositories.pipeline import (
    advance_checkpoint,
    advance_stage,
    checkpoint_key_for,
    finish_run,
    get_checkpoint,
    get_or_create_status,
    record_attempt_failure,
    record_dead_letter,
    start_run,
)

__all__ = [
    "UpsertOutcome",
    "UpsertResult",
    "advance_checkpoint",
    "advance_stage",
    "checkpoint_key_for",
    "count_unindexed_chunks",
    "finish_run",
    "get_checkpoint",
    "get_or_create_authors",
    "get_or_create_status",
    "mark_chunks_indexed",
    "normalize_author_name",
    "record_attempt_failure",
    "record_dead_letter",
    "replace_chunks",
    "save_document_text",
    "start_run",
    "upsert_paper",
]
