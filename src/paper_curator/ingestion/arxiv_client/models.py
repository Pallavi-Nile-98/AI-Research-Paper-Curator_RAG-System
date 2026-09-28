"""Validated representations of arXiv API responses.

These are the boundary between an untrusted external feed and the rest of the
application. Everything downstream — deduplication, PDF download, chunking —
assumes an :class:`ArxivPaper` is well formed, so validation happens here, once,
and a malformed entry fails loudly at the edge rather than producing a confusing
error three stages later.

Pydantic rather than a dataclass for exactly that reason: external input gets
validated, internal data structures do not need to be.
"""

from __future__ import annotations

import datetime as dt
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Modern identifiers are YYMM.NNNNN with four or five digits after the dot.
# Pre-2007 identifiers look like "cs/0501001" or "math.GT/0309136".
ARXIV_ID_PATTERN = re.compile(
    r"""
    ^(?:
        \d{4}\.\d{4,5}                 # 2401.12345
      | [a-z-]+(?:\.[A-Z]{2})?/\d{7}   # cs/0501001, math.GT/0309136
    )$
    """,
    re.VERBOSE,
)


class ArxivPaper(BaseModel):
    """One version of one paper, as returned by the arXiv API.

    Frozen because it represents a response already received: mutating it would
    only ever hide a bug. ``extra="forbid"`` means a typo in a field name is a
    validation error rather than a silently ignored attribute.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    arxiv_id: str = Field(description="Identifier without version, e.g. 2401.12345")
    version: int = Field(ge=1, description="1-based version number")

    title: str = Field(min_length=1)
    abstract: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1, description="Ordered; first author first")

    published_at: dt.datetime = Field(description="When version 1 appeared")
    updated_at: dt.datetime = Field(description="When THIS version appeared")

    primary_category: str = Field(min_length=1)
    categories: list[str] = Field(min_length=1)

    doi: str | None = None
    journal_ref: str | None = None
    comment: str | None = None

    abs_url: str
    pdf_url: str

    @field_validator("arxiv_id")
    @classmethod
    def _validate_arxiv_id(cls, value: str) -> str:
        if not ARXIV_ID_PATTERN.match(value):
            msg = f"not a recognised arXiv identifier: {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("published_at", "updated_at")
    @classmethod
    def _require_timezone(cls, value: dt.datetime) -> dt.datetime:
        """Reject naive datetimes.

        A naive timestamp compared against a timezone-aware checkpoint raises at
        runtime, and incremental sync compares these on every run.
        """
        if value.tzinfo is None:
            msg = "timestamp must be timezone-aware"
            raise ValueError(msg)
        return value

    @field_validator("categories")
    @classmethod
    def _primary_category_first(cls, value: list[str]) -> list[str]:
        """Deduplicate while preserving order."""
        seen: set[str] = set()
        ordered: list[str] = []
        for category in value:
            if category not in seen:
                seen.add(category)
                ordered.append(category)
        return ordered

    @property
    def versioned_id(self) -> str:
        """Identifier including version, e.g. ``2401.12345v2``."""
        return f"{self.arxiv_id}v{self.version}"

    def __str__(self) -> str:
        return self.versioned_id


class ArxivPage(BaseModel):
    """One page of results, with the counters needed to drive pagination.

    ``total_results`` is what makes bounded paging possible: without it a client
    cannot distinguish "this page is empty because we reached the end" from
    "this page is empty because the API hiccupped", and arXiv does
    intermittently return empty pages for valid queries.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    papers: list[ArxivPaper]
    total_results: int = Field(ge=0, description="Matches across the whole query")
    start_index: int = Field(ge=0, description="Offset of this page")
    items_per_page: int = Field(ge=0)

    @property
    def is_empty(self) -> bool:
        return len(self.papers) == 0

    @property
    def next_start_index(self) -> int:
        """Offset for the following page."""
        return self.start_index + len(self.papers)

    @property
    def has_more(self) -> bool:
        """True when more results exist beyond this page.

        Based on the reported total rather than on page fullness. arXiv can
        return a short page mid-result-set, and treating that as the end would
        silently truncate ingestion.
        """
        return self.next_start_index < self.total_results
