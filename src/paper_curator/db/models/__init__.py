"""ORM models for the application.

Importing this package registers every model on ``Base.metadata``. Alembic's
autogenerate relies on that: a model in a module nobody imports is invisible to
it, and the resulting migration silently omits the table.
"""

from paper_curator.db.models.feedback import UserFeedback, UserQuery
from paper_curator.db.models.paper import (
    Author,
    Chunk,
    DocumentText,
    Paper,
    PaperAuthor,
)
from paper_curator.db.models.pipeline import (
    DocumentProcessingStatus,
    FailedDocument,
    IngestionRun,
    PipelineCheckpoint,
)

__all__ = [
    "Author",
    "Chunk",
    "DocumentProcessingStatus",
    "DocumentText",
    "FailedDocument",
    "IngestionRun",
    "Paper",
    "PaperAuthor",
    "PipelineCheckpoint",
    "UserFeedback",
    "UserQuery",
]
