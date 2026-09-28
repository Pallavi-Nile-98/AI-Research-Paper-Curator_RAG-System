"""Tests for parsing the arXiv Atom feed.

Parsing is the boundary between an untrusted external feed and everything
downstream, so the cases here are the ones that would otherwise corrupt data
quietly: versions merged into one paper, ragged whitespace in citations, and
API errors that arrive with HTTP 200 and parse as a nonsensical paper.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from paper_curator.ingestion.arxiv_client.errors import ArxivParseError
from paper_curator.ingestion.arxiv_client.parser import parse_feed, split_versioned_id

FIXTURES = Path(__file__).parent.parent / "fixtures" / "arxiv"


@pytest.fixture
def two_entry_feed() -> str:
    return (FIXTURES / "two_entries.xml").read_text(encoding="utf-8")


def _feed_around(entry_xml: str, *, total: int = 1) -> str:
    """Wrap a single entry in a minimal but valid Atom feed."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>{total}</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>1</opensearch:itemsPerPage>
  {entry_xml}
</feed>"""


MINIMAL_ENTRY = """
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <updated>2024-01-01T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>A Title</title>
    <summary>An abstract.</summary>
    <author><name>Solo Author</name></author>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
  </entry>
"""


@pytest.mark.unit
class TestSplitVersionedId:
    @pytest.mark.parametrize(
        ("url", "expected_id", "expected_version"),
        [
            ("http://arxiv.org/abs/2401.12345v2", "2401.12345", 2),
            ("https://arxiv.org/abs/2401.12345v1", "2401.12345", 1),
            ("http://arxiv.org/abs/2401.1234v11", "2401.1234", 11),
            # Pre-2007 identifiers carry a slash and an archive prefix.
            ("http://arxiv.org/abs/cs/0501001v1", "cs/0501001", 1),
            ("http://arxiv.org/abs/math.GT/0309136v2", "math.GT/0309136", 2),
        ],
    )
    def test_splits_identifier_from_version(
        self, url: str, expected_id: str, expected_version: int
    ) -> None:
        assert split_versioned_id(url) == (expected_id, expected_version)

    def test_defaults_to_version_one_without_a_suffix(self) -> None:
        """Humans paste unversioned identifiers; the API never sends them."""
        assert split_versioned_id("http://arxiv.org/abs/2401.12345") == ("2401.12345", 1)

    def test_rejects_a_non_arxiv_url(self) -> None:
        with pytest.raises(ArxivParseError, match="not an arXiv abstract URL"):
            split_versioned_id("https://example.com/paper/1")


@pytest.mark.unit
class TestParseFeed:
    def test_parses_every_entry(self, two_entry_feed: str) -> None:
        page = parse_feed(two_entry_feed)
        assert len(page.papers) == 2

    def test_reads_pagination_counters(self, two_entry_feed: str) -> None:
        """total_results is what distinguishes 'end of results' from 'API hiccup'."""
        page = parse_feed(two_entry_feed)
        assert page.total_results == 137
        assert page.start_index == 0
        assert page.items_per_page == 2
        assert page.has_more is True
        assert page.next_start_index == 2

    def test_separates_identifier_from_version(self, two_entry_feed: str) -> None:
        """Without this, every revision looks like a brand new paper."""
        first = parse_feed(two_entry_feed).papers[0]
        assert first.arxiv_id == "2401.12345"
        assert first.version == 2
        assert first.versioned_id == "2401.12345v2"

    def test_normalises_hard_wrapped_title(self, two_entry_feed: str) -> None:
        """Titles arrive hard-wrapped across lines, with indentation."""
        first = parse_feed(two_entry_feed).papers[0]
        assert first.title == (
            "Hybrid Retrieval for Scientific Question Answering: A Systematic Study"
        )
        assert "\n" not in first.title

    def test_normalises_hard_wrapped_abstract(self, two_entry_feed: str) -> None:
        first = parse_feed(two_entry_feed).papers[0]
        assert "\n" not in first.abstract
        assert first.abstract.startswith("We study the combination")
        assert first.abstract.endswith("across six query categories.")

    def test_preserves_author_order(self, two_entry_feed: str) -> None:
        """First and last author are not interchangeable in academic publishing."""
        first = parse_feed(two_entry_feed).papers[0]
        assert first.authors == ["Jane A. Doe", "Rahul Mehta", "Wei Zhang"]

    def test_primary_category_is_listed_first(self, two_entry_feed: str) -> None:
        """The feed lists categories alphabetically, not primary-first."""
        first = parse_feed(two_entry_feed).papers[0]
        assert first.primary_category == "cs.IR"
        assert first.categories[0] == "cs.IR"
        assert set(first.categories) == {"cs.CL", "cs.IR", "cs.LG"}

    def test_parses_timestamps_as_timezone_aware(self, two_entry_feed: str) -> None:
        first = parse_feed(two_entry_feed).papers[0]
        assert first.published_at == dt.datetime(2024, 1, 20, 9, 15, tzinfo=dt.UTC)
        assert first.updated_at == dt.datetime(2024, 2, 1, 10, 30, tzinfo=dt.UTC)
        assert first.updated_at.tzinfo is not None

    def test_parses_optional_metadata(self, two_entry_feed: str) -> None:
        first = parse_feed(two_entry_feed).papers[0]
        assert first.doi == "10.1000/example.2401.12345"
        assert first.journal_ref == "Journal of Examples 12 (2024) 1-14"
        assert first.comment == "14 pages, 5 figures. Accepted at an example venue"

    def test_absent_optional_metadata_is_none(self, two_entry_feed: str) -> None:
        second = parse_feed(two_entry_feed).papers[1]
        assert second.doi is None
        assert second.journal_ref is None
        assert second.comment is None

    def test_extracts_both_links(self, two_entry_feed: str) -> None:
        first = parse_feed(two_entry_feed).papers[0]
        assert first.abs_url == "http://arxiv.org/abs/2401.12345v2"
        assert first.pdf_url == "http://arxiv.org/pdf/2401.12345v2"

    def test_derives_pdf_url_when_the_link_is_missing(self) -> None:
        page = parse_feed(_feed_around(MINIMAL_ENTRY))
        assert page.papers[0].pdf_url == "https://arxiv.org/pdf/2401.00001v1"


@pytest.mark.unit
class TestMalformedInput:
    def test_rejects_input_that_is_not_xml(self) -> None:
        with pytest.raises(ArxivParseError, match="not well-formed XML"):
            parse_feed("<feed><unclosed>")

    def test_treats_an_arxiv_error_entry_as_a_failure(self) -> None:
        """Query errors are reported as a normal feed with HTTP 200.

        Parsing that as a paper yields a nonsense record instead of a clear
        failure, so the error id is detected explicitly.
        """
        error_entry = """
          <entry>
            <id>http://arxiv.org/api/errors#incorrect_id_format</id>
            <title>Error</title>
            <summary>incorrect id format for abc</summary>
            <updated>2024-01-01T00:00:00Z</updated>
            <published>2024-01-01T00:00:00Z</published>
            <author><name>arXiv api core</name></author>
          </entry>
        """
        with pytest.raises(ArxivParseError, match="arXiv returned an error entry"):
            parse_feed(_feed_around(error_entry), strict=True)

    @pytest.mark.parametrize(
        ("removed", "description"),
        [
            ("<author><name>Solo Author</name></author>", "no authors"),
            ("<title>A Title</title>", "no title"),
            ("<summary>An abstract.</summary>", "no summary"),
        ],
    )
    def test_strict_mode_rejects_an_entry_missing_a_required_field(
        self, removed: str, description: str
    ) -> None:
        broken = MINIMAL_ENTRY.replace(removed, "")
        with pytest.raises(ArxivParseError):
            parse_feed(_feed_around(broken), strict=True)

    def test_lenient_mode_skips_a_bad_entry_and_keeps_the_rest(self, two_entry_feed: str) -> None:
        """A nightly job should make progress, not halt on one malformed record."""
        damaged = two_entry_feed.replace(
            "<author>\n      <name>Priya Sharma</name>\n    </author>", ""
        )
        page = parse_feed(damaged, strict=False)
        assert len(page.papers) == 1
        assert page.papers[0].arxiv_id == "2401.12345"

    def test_falls_back_to_the_first_category_when_primary_is_absent(self) -> None:
        """Some older entries omit arxiv:primary_category entirely."""
        without_primary = MINIMAL_ENTRY.replace('<arxiv:primary_category term="cs.CL"/>', "")
        page = parse_feed(_feed_around(without_primary))
        assert page.papers[0].primary_category == "cs.CL"


@pytest.mark.unit
class TestEmptyFeed:
    def test_an_empty_feed_parses_to_an_empty_page(self) -> None:
        empty = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>0</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>0</opensearch:itemsPerPage>
</feed>"""
        page = parse_feed(empty)
        assert page.is_empty
        assert page.has_more is False
