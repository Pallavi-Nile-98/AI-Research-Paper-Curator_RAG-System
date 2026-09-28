"""Builder for arXiv API search queries.

The arXiv API takes a single ``search_query`` string with its own small syntax:
field prefixes (``cat:``, ``all:``, ``au:``, ``ti:``), the boolean operators
``AND``, ``OR`` and ``ANDNOT``, and date ranges in
``field:[YYYYMMDDHHMM TO YYYYMMDDHHMM]`` form.

Building that string by concatenation at call sites goes wrong quickly — an
unquoted multi-word phrase silently becomes two terms, and precedence between
AND and OR is easy to get backwards. This module builds it in one place, with
tests.

Reference: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

# Category identifiers: an archive, optionally a dot and a subject class.
# e.g. cs.CL, math.GT, hep-th, econ.EM
CATEGORY_PATTERN = re.compile(r"^[a-z][a-z-]*(?:\.[A-Za-z]{2,})?$")

# arXiv's date-range literal format.
_ARXIV_DATE_FORMAT = "%Y%m%d%H%M"


class SortBy(StrEnum):
    """Field the API sorts results on."""

    RELEVANCE = "relevance"
    LAST_UPDATED = "lastUpdatedDate"
    SUBMITTED = "submittedDate"


class SortOrder(StrEnum):
    """Direction the API sorts results in."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


def _quote_phrase(value: str) -> str:
    """Wrap a term in quotes if it contains whitespace.

    Without this, ``all:information retrieval`` is parsed as ``all:information``
    followed by a bare ``retrieval`` — which matches far more than intended and
    is hard to notice, because results still come back.
    """
    cleaned = value.strip().replace('"', "")
    if not cleaned:
        msg = "search term must not be empty"
        raise ValueError(msg)
    return f'"{cleaned}"' if " " in cleaned else cleaned


def _format_date(value: dt.datetime) -> str:
    """Render a datetime in arXiv's range format, normalised to UTC."""
    if value.tzinfo is None:
        msg = "date bounds must be timezone-aware"
        raise ValueError(msg)
    return value.astimezone(dt.UTC).strftime(_ARXIV_DATE_FORMAT)


@dataclass(frozen=True, slots=True)
class ArxivQuery:
    """A search over the arXiv API.

    Clauses combine as: categories OR'd together, keywords OR'd together,
    authors OR'd together, and those groups AND'd with each other and with any
    date range. Parenthesising each group is what keeps AND/OR precedence from
    quietly inverting the meaning.

    ``updated_after`` drives incremental synchronisation. It filters on the
    *updated* timestamp rather than the submitted one, so a 2019 paper revised
    yesterday is picked up — watermarking on submission date would miss every
    revision to older work.
    """

    categories: Sequence[str] = field(default_factory=tuple)
    keywords: Sequence[str] = field(default_factory=tuple)
    authors: Sequence[str] = field(default_factory=tuple)

    submitted_after: dt.datetime | None = None
    submitted_before: dt.datetime | None = None
    updated_after: dt.datetime | None = None
    updated_before: dt.datetime | None = None

    sort_by: SortBy = SortBy.LAST_UPDATED
    # Ascending by default, because incremental sync advances a high-water mark
    # and must process oldest-first for a partial run to leave a usable
    # checkpoint. Descending would mean a crash halfway leaves a gap.
    sort_order: SortOrder = SortOrder.ASCENDING

    def __post_init__(self) -> None:
        for category in self.categories:
            if not CATEGORY_PATTERN.match(category):
                msg = f"not a valid arXiv category: {category!r}"
                raise ValueError(msg)

        if (
            self.submitted_after
            and self.submitted_before
            and self.submitted_after > self.submitted_before
        ):
            msg = "submitted_after must not be later than submitted_before"
            raise ValueError(msg)

        if self.updated_after and self.updated_before and self.updated_after > self.updated_before:
            msg = "updated_after must not be later than updated_before"
            raise ValueError(msg)

        if not any(
            (
                self.categories,
                self.keywords,
                self.authors,
                self.submitted_after,
                self.updated_after,
            )
        ):
            msg = "query must constrain at least one of: categories, keywords, authors, dates"
            raise ValueError(msg)

    def _date_clause(
        self,
        field_name: str,
        after: dt.datetime | None,
        before: dt.datetime | None,
    ) -> str | None:
        """Build one ``field:[lower TO upper]`` clause, or None if unbounded.

        arXiv has no open-ended range syntax, so a missing bound is replaced
        with a sentinel: the dawn of the archive for a lower bound, and a far
        future date for an upper one.
        """
        if after is None and before is None:
            return None
        lower = _format_date(after) if after else "199101010000"
        upper = _format_date(before) if before else "999912312359"
        return f"{field_name}:[{lower} TO {upper}]"

    def to_search_query(self) -> str:
        """Render the ``search_query`` parameter value."""
        groups: list[str] = []

        if self.categories:
            groups.append("(" + " OR ".join(f"cat:{c}" for c in self.categories) + ")")
        if self.keywords:
            groups.append("(" + " OR ".join(f"all:{_quote_phrase(k)}" for k in self.keywords) + ")")
        if self.authors:
            groups.append("(" + " OR ".join(f"au:{_quote_phrase(a)}" for a in self.authors) + ")")

        submitted = self._date_clause("submittedDate", self.submitted_after, self.submitted_before)
        if submitted:
            groups.append(submitted)

        updated = self._date_clause("lastUpdatedDate", self.updated_after, self.updated_before)
        if updated:
            groups.append(updated)

        return " AND ".join(groups)

    def to_params(self, *, start: int, max_results: int) -> dict[str, str | int]:
        """Render the complete query-string parameters for one page."""
        if start < 0:
            msg = f"start must be non-negative, got {start}"
            raise ValueError(msg)
        if max_results < 1:
            msg = f"max_results must be at least 1, got {max_results}"
            raise ValueError(msg)

        return {
            "search_query": self.to_search_query(),
            "start": start,
            "max_results": max_results,
            "sortBy": self.sort_by.value,
            "sortOrder": self.sort_order.value,
        }
