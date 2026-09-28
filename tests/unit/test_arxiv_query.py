"""Tests for the arXiv search-query builder.

The failure mode this guards against is silent: a badly built query still
returns results, just the wrong ones. Nothing errors, so it is only noticed
later when retrieval quality looks inexplicably poor.
"""

from __future__ import annotations

import datetime as dt

import pytest

from paper_curator.ingestion.arxiv_client.query import ArxivQuery, SortBy, SortOrder


@pytest.mark.unit
class TestClauseConstruction:
    def test_single_category(self) -> None:
        assert ArxivQuery(categories=["cs.CL"]).to_search_query() == "(cat:cs.CL)"

    def test_categories_are_combined_with_or(self) -> None:
        """A paper in any requested category should match, not only one in all."""
        query = ArxivQuery(categories=["cs.CL", "cs.IR"])
        assert query.to_search_query() == "(cat:cs.CL OR cat:cs.IR)"

    def test_groups_are_combined_with_and(self) -> None:
        query = ArxivQuery(categories=["cs.CL"], keywords=["retrieval"])
        assert query.to_search_query() == "(cat:cs.CL) AND (all:retrieval)"

    def test_each_group_is_parenthesised(self) -> None:
        """Without parentheses, AND binds tighter than OR and inverts the meaning.

        `cat:a OR cat:b AND all:x` means `cat:a OR (cat:b AND all:x)` -- which
        returns everything in category a regardless of the keyword.
        """
        query = ArxivQuery(categories=["cs.CL", "cs.IR"], keywords=["bm25", "dense"])
        rendered = query.to_search_query()
        assert rendered == "(cat:cs.CL OR cat:cs.IR) AND (all:bm25 OR all:dense)"

    def test_multi_word_keywords_are_quoted(self) -> None:
        """Unquoted, `all:information retrieval` parses as two separate terms."""
        query = ArxivQuery(keywords=["information retrieval"])
        assert query.to_search_query() == '(all:"information retrieval")'

    def test_single_word_keywords_are_not_quoted(self) -> None:
        assert ArxivQuery(keywords=["transformer"]).to_search_query() == "(all:transformer)"

    def test_author_names_use_the_author_prefix(self) -> None:
        query = ArxivQuery(authors=["Jane Doe"])
        assert query.to_search_query() == '(au:"Jane Doe")'

    def test_embedded_quotes_are_stripped(self) -> None:
        """A stray quote would otherwise terminate the phrase early."""
        query = ArxivQuery(keywords=['neural "search" systems'])
        assert query.to_search_query() == '(all:"neural search systems")'


@pytest.mark.unit
class TestDateRanges:
    def test_updated_after_produces_an_open_ended_upper_bound(self) -> None:
        query = ArxivQuery(
            categories=["cs.CL"],
            updated_after=dt.datetime(2024, 1, 15, 12, 30, tzinfo=dt.UTC),
        )
        assert "lastUpdatedDate:[202401151230 TO 999912312359]" in query.to_search_query()

    def test_submitted_range_uses_both_bounds(self) -> None:
        query = ArxivQuery(
            submitted_after=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
            submitted_before=dt.datetime(2024, 1, 31, 23, 59, tzinfo=dt.UTC),
        )
        assert "submittedDate:[202401010000 TO 202401312359]" in query.to_search_query()

    def test_bounds_are_normalised_to_utc(self) -> None:
        """A local-time bound would shift the window by the offset."""
        eastern = dt.timezone(dt.timedelta(hours=-5))
        query = ArxivQuery(
            categories=["cs.CL"],
            updated_after=dt.datetime(2024, 1, 15, 7, 30, tzinfo=eastern),
        )
        assert "lastUpdatedDate:[202401151230 TO" in query.to_search_query()

    def test_naive_datetimes_are_rejected(self) -> None:
        query = ArxivQuery(
            categories=["cs.CL"],
            updated_after=dt.datetime(2024, 1, 15, 12, 30),  # intentionally naive
        )
        with pytest.raises(ValueError, match="timezone-aware"):
            query.to_search_query()


@pytest.mark.unit
class TestValidation:
    @pytest.mark.parametrize("category", ["cs.CL", "math.GT", "hep-th", "econ.EM"])
    def test_accepts_valid_categories(self, category: str) -> None:
        assert ArxivQuery(categories=[category]).to_search_query()

    @pytest.mark.parametrize("category", ["CS.CL", "cs..CL", "cs/CL", "", "cs.C"])
    def test_rejects_malformed_categories(self, category: str) -> None:
        with pytest.raises(ValueError, match="not a valid arXiv category"):
            ArxivQuery(categories=[category])

    def test_rejects_a_query_with_no_constraints(self) -> None:
        """An unconstrained query would page through the entire archive."""
        with pytest.raises(ValueError, match="must constrain at least one"):
            ArxivQuery()

    def test_rejects_an_inverted_date_range(self) -> None:
        with pytest.raises(ValueError, match="must not be later than"):
            ArxivQuery(
                updated_after=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
                updated_before=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
            )


@pytest.mark.unit
class TestParameters:
    def test_defaults_suit_incremental_sync(self) -> None:
        """Ascending by last-updated is what makes a partial run resumable.

        Descending would mean a crash halfway through leaves a gap between the
        checkpoint and the records already processed.
        """
        query = ArxivQuery(categories=["cs.CL"])
        assert query.sort_by is SortBy.LAST_UPDATED
        assert query.sort_order is SortOrder.ASCENDING

    def test_renders_all_request_parameters(self) -> None:
        params = ArxivQuery(categories=["cs.CL"]).to_params(start=20, max_results=50)
        assert params == {
            "search_query": "(cat:cs.CL)",
            "start": 20,
            "max_results": 50,
            "sortBy": "lastUpdatedDate",
            "sortOrder": "ascending",
        }

    @pytest.mark.parametrize(("start", "max_results"), [(-1, 10), (0, 0), (0, -5)])
    def test_rejects_invalid_paging_parameters(self, start: int, max_results: int) -> None:
        query = ArxivQuery(categories=["cs.CL"])
        with pytest.raises(ValueError, match="must be"):
            query.to_params(start=start, max_results=max_results)

    def test_is_immutable(self) -> None:
        """Frozen so a query cannot be mutated between pages of one iteration."""
        query = ArxivQuery(categories=["cs.CL"])
        with pytest.raises(AttributeError):
            query.categories = ["cs.IR"]  # type: ignore[misc]
