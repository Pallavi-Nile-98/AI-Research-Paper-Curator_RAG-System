"""Database layer: ORM models, engine and session management.

Depends only on :mod:`paper_curator.core`. Nothing here imports from
``ingestion``, ``retrieval``, ``generation`` or ``api`` — dependencies point
inward, per ADR-0005.
"""

from paper_curator.db.base import Base, TimestampMixin
from paper_curator.db.enums import (
    ExtractionMethod,
    FeedbackRating,
    ProcessingStage,
    RetrievalMode,
    RunStatus,
    RunType,
)
from paper_curator.db.models import (
    Author,
    Chunk,
    DocumentProcessingStatus,
    DocumentText,
    FailedDocument,
    IngestionRun,
    Paper,
    PaperAuthor,
    PipelineCheckpoint,
    UserFeedback,
    UserQuery,
)
from paper_curator.db.session import (
    dispose_engine,
    get_db_session,
    get_engine,
    get_session_factory,
    session_scope,
)

__all__ = [
    "Author",
    "Base",
    "Chunk",
    "DocumentProcessingStatus",
    "DocumentText",
    "ExtractionMethod",
    "FailedDocument",
    "FeedbackRating",
    "IngestionRun",
    "Paper",
    "PaperAuthor",
    "PipelineCheckpoint",
    "ProcessingStage",
    "RetrievalMode",
    "RunStatus",
    "RunType",
    "TimestampMixin",
    "UserFeedback",
    "UserQuery",
    "dispose_engine",
    "get_db_session",
    "get_engine",
    "get_session_factory",
    "session_scope",
]
