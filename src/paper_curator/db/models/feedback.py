"""Models for user queries and the feedback given on their answers.

These two tables exist to answer one question: *which configurations actually
produce good answers?* That requires recording not only what was asked and
answered, but the exact prompt version, model, and retrieval settings behind it.
Feedback with no configuration attached tells you an answer was bad without
telling you what to change.

Privacy note. Everything here is user-generated, unlike papers and chunks which
are derived from public sources. Two consequences are enforced by design:

* No user identifier is stored. There is no account system (Phase 3 deliberately
  ships without authentication), and adding a fingerprint or IP address would
  create a tracking dataset the project has no use for.
* Free-text comments are stored here in PostgreSQL and are **not** sent to the
  hosted observability service, unlike query text — see ADR-0007. A comment box
  is where a user is most likely to paste something personal.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Enum,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from paper_curator.db.base import Base, TimestampMixin
from paper_curator.db.enums import FeedbackRating, RetrievalMode


class UserQuery(Base, TimestampMixin):
    """One question asked of the system, and the answer produced.

    The configuration columns turn this table into an experiment log. Because
    ``prompt_version``, ``model_name``, ``retrieval_mode`` and
    ``retrieval_config`` are recorded per query, a later analysis can compare
    negative-feedback rates across prompt versions, or latency across retrieval
    modes, without any extra instrumentation.

    Latency is stored per stage rather than as a single total. On a CPU-only
    machine generation dominates end-to-end time (ADR-0003), so a single number
    would hide whether retrieval is fast or slow.
    """

    __tablename__ = "user_queries"
    __table_args__ = (
        # The correlation ID that ties this row to structured logs and to a
        # Langfuse trace. Unique so a retried client request cannot duplicate it.
        UniqueConstraint("request_id", name="uq_user_queries_request_id"),
        Index("ix_user_queries_created_at", "created_at"),
        Index("ix_user_queries_retrieval_mode", "retrieval_mode"),
        Index("ix_user_queries_prompt_version", "prompt_version"),
        CheckConstraint("top_k > 0", name="top_k_positive"),
        CheckConstraint("total_ms >= 0", name="total_ms_non_negative"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)

    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str | None] = mapped_column(
        Text, doc="NULL when generation failed or evidence was insufficient"
    )
    insufficient_evidence: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        doc="True when the system declined to answer rather than guessing",
    )

    # --- Configuration under which this answer was produced -------------------
    retrieval_mode: Mapped[RetrievalMode] = mapped_column(
        Enum(RetrievalMode, native_enum=False, length=16), nullable=False
    )
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    reranker_enabled: Mapped[bool] = mapped_column(nullable=False, default=False)
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    model_name: Mapped[str | None] = mapped_column(String(128))
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    # Fusion weights, candidate pool size, active metadata filters. JSONB rather
    # than columns because these parameters change as retrieval is tuned, and a
    # migration per experiment would be absurd.
    retrieval_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # --- Outcome --------------------------------------------------------------
    cited_papers: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="arXiv IDs actually cited in the answer, for citation-accuracy analysis",
    )
    chunks_retrieved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)

    # --- Latency, per stage ---------------------------------------------------
    retrieval_ms: Mapped[int | None] = mapped_column(Integer)
    rerank_ms: Mapped[int | None] = mapped_column(Integer)
    generation_ms: Mapped[int | None] = mapped_column(Integer)
    total_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    error_type: Mapped[str | None] = mapped_column(
        String(128), doc="Set when the request failed; NULL on success"
    )

    feedback: Mapped[UserFeedback | None] = relationship(
        back_populates="query", cascade="all, delete-orphan", uselist=False
    )

    def __repr__(self) -> str:
        return f"<UserQuery {self.request_id} mode={self.retrieval_mode}>"


class UserFeedback(Base, TimestampMixin):
    """A user's rating of one answer.

    One row per query, so submitting feedback twice updates the existing
    judgement rather than accumulating duplicates that would skew any rate
    computed from this table.

    The rating is binary by design — see :class:`~paper_curator.db.enums.FeedbackRating`.
    Configuration is deliberately *not* duplicated here: it lives on the
    referenced :class:`UserQuery`, so there is exactly one place where the
    conditions of an answer are recorded and no way for the two to disagree.
    """

    __tablename__ = "user_feedback"
    __table_args__ = (
        UniqueConstraint("query_id", name="uq_user_feedback_query_id"),
        Index("ix_user_feedback_rating", "rating"),
        Index("ix_user_feedback_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    query_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("user_queries.id", ondelete="CASCADE"), nullable=False
    )

    rating: Mapped[FeedbackRating] = mapped_column(
        Enum(FeedbackRating, native_enum=False, length=16), nullable=False
    )
    comment: Mapped[str | None] = mapped_column(
        Text,
        doc="Optional free text. Treated as potentially sensitive: never sent to "
        "external tracing, and excluded from exports unless explicitly requested.",
    )

    query: Mapped[UserQuery] = relationship(back_populates="feedback")

    def __repr__(self) -> str:
        return f"<UserFeedback query={self.query_id} {self.rating}>"
