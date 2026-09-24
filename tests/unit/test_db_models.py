"""Schema invariant tests for the ORM models.

These run against ``Base.metadata`` rather than a live database, so they need no
Postgres and run in milliseconds. That is the point: the constraints asserted
here are the ones the pipeline's correctness depends on, and a regression should
fail in a unit test rather than during an ingestion run.

Integration tests that exercise real inserts against PostgreSQL are separate and
marked ``integration``.
"""

from __future__ import annotations

import pytest
from sqlalchemy import DateTime, UniqueConstraint

from paper_curator.db.base import NAMING_CONVENTION, Base
from paper_curator.db.enums import (
    ExtractionMethod,
    FeedbackRating,
    ProcessingStage,
    RetrievalMode,
    RunStatus,
    RunType,
)
from paper_curator.db.models import (
    Chunk,
    DocumentProcessingStatus,
    Paper,
    PaperAuthor,
)

EXPECTED_TABLES = {
    "authors",
    "chunks",
    "document_processing_status",
    "document_texts",
    "failed_documents",
    "ingestion_runs",
    "paper_authors",
    "papers",
    "pipeline_checkpoints",
    "user_feedback",
    "user_queries",
}


def _unique_column_sets(table_name: str) -> set[frozenset[str]]:
    """Return every unique constraint and unique index as a set of column names."""
    table = Base.metadata.tables[table_name]
    sets: set[frozenset[str]] = set()
    for constraint in table.constraints:
        if isinstance(constraint, UniqueConstraint):
            sets.add(frozenset(col.name for col in constraint.columns))
    for index in table.indexes:
        if index.unique:
            sets.add(frozenset(col.name for col in index.columns))
    return sets


@pytest.mark.unit
class TestSchemaRegistration:
    def test_all_expected_tables_are_registered(self) -> None:
        """A model in a module nobody imports is invisible to Alembic autogenerate."""
        assert set(Base.metadata.tables) == EXPECTED_TABLES

    def test_naming_convention_is_applied(self) -> None:
        assert Base.metadata.naming_convention == NAMING_CONVENTION


@pytest.mark.unit
class TestIdempotencyConstraints:
    """The constraints that make re-running ingestion safe.

    Each of these is the database-level guarantee behind a claim made in
    ADR-0002. If one is dropped, duplicate rows become possible and the only
    thing preventing them is application code being correct on every path.
    """

    def test_papers_are_unique_per_arxiv_id_and_version(self) -> None:
        """Re-ingesting the same paper version must be rejected by the database."""
        assert frozenset({"arxiv_id", "version"}) in _unique_column_sets("papers")

    def test_only_one_version_per_paper_can_be_latest(self) -> None:
        """A partial unique index enforces this rather than application logic."""
        papers = Base.metadata.tables["papers"]
        partial = [
            index
            for index in papers.indexes
            if index.unique and index.dialect_options["postgresql"].get("where") is not None
        ]
        assert partial, "expected a partial unique index guarding is_latest"
        assert {col.name for col in partial[0].columns} == {"arxiv_id"}

    def test_chunk_opensearch_id_is_unique(self) -> None:
        """Deterministic document IDs are what make re-indexing overwrite, not duplicate."""
        assert frozenset({"opensearch_doc_id"}) in _unique_column_sets("chunks")

    def test_chunks_are_unique_per_paper_and_index(self) -> None:
        assert frozenset({"paper_id", "chunk_index"}) in _unique_column_sets("chunks")

    def test_checkpoint_key_is_unique(self) -> None:
        """Two rows for one stream would make the resume point ambiguous."""
        assert frozenset({"checkpoint_key"}) in _unique_column_sets("pipeline_checkpoints")

    def test_one_processing_status_row_per_paper(self) -> None:
        assert frozenset({"paper_id"}) in _unique_column_sets("document_processing_status")

    def test_request_id_is_unique(self) -> None:
        """Correlation IDs tie a query row to its logs and trace; duplicates break that."""
        assert frozenset({"request_id"}) in _unique_column_sets("user_queries")

    def test_one_feedback_row_per_query(self) -> None:
        """Resubmitting feedback updates the judgement instead of skewing any rate."""
        assert frozenset({"query_id"}) in _unique_column_sets("user_feedback")


@pytest.mark.unit
class TestCascadeBehaviour:
    """Deleting a paper must not leave orphaned derived rows behind."""

    @pytest.mark.parametrize(
        ("table_name", "column_name"),
        [
            ("chunks", "paper_id"),
            ("document_texts", "paper_id"),
            ("paper_authors", "paper_id"),
            ("document_processing_status", "paper_id"),
        ],
    )
    def test_derived_rows_cascade_from_papers(self, table_name: str, column_name: str) -> None:
        table = Base.metadata.tables[table_name]
        fks = [fk for fk in table.foreign_keys if fk.parent.name == column_name]
        assert fks, f"{table_name}.{column_name} should be a foreign key"
        assert fks[0].ondelete == "CASCADE"

    def test_failed_documents_survive_paper_deletion(self) -> None:
        """A failure record that vanishes with its paper is useless for analysis.

        failed_documents denormalises arxiv_id precisely so the record remains
        meaningful after the paper row is gone, so the FK nulls rather than
        cascading.
        """
        table = Base.metadata.tables["failed_documents"]
        fks = [fk for fk in table.foreign_keys if fk.parent.name == "paper_id"]
        assert fks[0].ondelete == "SET NULL"
        assert not table.columns["arxiv_id"].nullable


@pytest.mark.unit
class TestColumnTypes:
    def test_every_timestamp_is_timezone_aware(self) -> None:
        """Naive timestamps silently discard offset information.

        A value written from one timezone then reads back wrong elsewhere, and
        the bug is invisible until someone compares two rows.
        """
        offenders = [
            f"{table.name}.{column.name}"
            for table in Base.metadata.tables.values()
            for column in table.columns
            if isinstance(column.type, DateTime) and not column.type.timezone
        ]
        assert offenders == []

    def test_chunk_indexed_at_is_nullable(self) -> None:
        """NULL indexed_at is the marker for dual-write reconciliation.

        A chunk committed to Postgres whose OpenSearch write failed is exactly a
        row with indexed_at IS NULL. Making the column NOT NULL would remove the
        only signal that the two stores disagree.
        """
        assert Base.metadata.tables["chunks"].columns["indexed_at"].nullable

    def test_author_position_is_not_nullable(self) -> None:
        """Author order is meaningful in academic publishing."""
        assert not Base.metadata.tables["paper_authors"].columns["position"].nullable


@pytest.mark.unit
class TestEnums:
    """Enum values are persisted as strings, so changing one is a data migration."""

    @pytest.mark.parametrize(
        ("enum_cls", "expected"),
        [
            (RunType, {"incremental", "backfill", "manual"}),
            (RunStatus, {"running", "succeeded", "failed", "partial"}),
            (
                ProcessingStage,
                {
                    "pending",
                    "downloaded",
                    "extracted",
                    "chunked",
                    "embedded",
                    "indexed",
                    "failed",
                    "skipped",
                },
            ),
            (ExtractionMethod, {"pymupdf", "ocr", "none"}),
            (RetrievalMode, {"keyword", "vector", "hybrid"}),
            (FeedbackRating, {"positive", "negative"}),
        ],
    )
    def test_enum_values(self, enum_cls: type[RunType], expected: set[str]) -> None:
        assert {member.value for member in enum_cls} == expected

    def test_enum_members_are_usable_as_plain_strings(self) -> None:
        """StrEnum keeps log output and JSON readable.

        Assigning through a ``str``-annotated name rather than comparing the
        member directly, because mypy narrows an enum member to a literal and
        flags direct comparison against a string as non-overlapping.
        """
        status: str = RunStatus.SUCCEEDED
        stage: str = ProcessingStage.INDEXED
        assert status == "succeeded"
        assert stage == "indexed"


@pytest.mark.unit
class TestModelBehaviour:
    """The small amount of logic that lives on the models."""

    def test_retries_exhausted_is_false_below_the_limit(self) -> None:
        status = DocumentProcessingStatus(paper_id=1, attempts=2, max_attempts=3)
        assert status.retries_exhausted is False

    def test_retries_exhausted_is_true_at_the_limit(self) -> None:
        status = DocumentProcessingStatus(paper_id=1, attempts=3, max_attempts=3)
        assert status.retries_exhausted is True

    def test_repr_identifies_the_row(self) -> None:
        """__repr__ appears in test failures and logs; it must identify the row."""
        assert "2401.12345v2" in repr(Paper(arxiv_id="2401.12345", version=2))
        assert "#3" in repr(Chunk(paper_id=1, chunk_index=3))
        assert repr(PaperAuthor(paper_id=1, author_id=2, position=0)) is not None
