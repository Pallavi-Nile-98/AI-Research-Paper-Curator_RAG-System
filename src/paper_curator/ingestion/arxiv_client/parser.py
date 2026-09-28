"""Parse the arXiv API's Atom XML into validated domain objects.

Uses ``defusedxml`` rather than the standard library's ``ElementTree``. The feed
is untrusted third-party input, and stdlib XML parsers are vulnerable to
entity-expansion attacks — the "billion laughs" pattern, where a few kilobytes
of XML expands to gigabytes in memory — and to external-entity references that
can read local files. ``defusedxml`` disables both. Ruff's flake8-bandit rules
flag the stdlib parsers for exactly this reason.

Three details of the format cause most of the bugs here:

* **Titles and abstracts are hard-wrapped**, with newlines and indentation
  inside the text. They need whitespace normalisation or every downstream
  comparison and every rendered citation carries ragged line breaks.
* **The version lives in the ``id`` URL**, not in its own field. ``2401.12345v2``
  must be split into an identifier and a version, or every revision is treated
  as a new paper.
* **Errors arrive as a normal feed** containing a single entry whose id points at
  ``arxiv.org/api/errors``, with HTTP 200. Parsing that as a paper produces a
  bizarre record instead of a clear failure.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Final
from xml.etree.ElementTree import Element

from defusedxml import ElementTree as DefusedElementTree

from paper_curator.ingestion.arxiv_client.errors import ArxivParseError
from paper_curator.ingestion.arxiv_client.models import ArxivPage, ArxivPaper

ATOM_NS: Final = "http://www.w3.org/2005/Atom"
ARXIV_NS: Final = "http://arxiv.org/schemas/atom"
OPENSEARCH_NS: Final = "http://a9.com/-/spec/opensearch/1.1/"

NAMESPACES: Final[dict[str, str]] = {
    "atom": ATOM_NS,
    "arxiv": ARXIV_NS,
    "opensearch": OPENSEARCH_NS,
}

# http://arxiv.org/abs/2401.12345v2  ->  ("2401.12345", "2")
# http://arxiv.org/abs/cs/0501001v1  ->  ("cs/0501001", "1")
_ABS_URL_PATTERN: Final = re.compile(
    r"^https?://arxiv\.org/abs/(?P<arxiv_id>.+?)(?:v(?P<version>\d+))?$"
)

_ERROR_ID_MARKER: Final = "arxiv.org/api/errors"


def _normalise_whitespace(value: str) -> str:
    """Collapse all runs of whitespace, including newlines, to single spaces."""
    return " ".join(value.split())


def _text_of(element: Element | None) -> str | None:
    """Return an element's normalised text, or None when absent or blank."""
    if element is None or element.text is None:
        return None
    normalised = _normalise_whitespace(element.text)
    return normalised or None


def _parse_timestamp(value: str, *, field_name: str) -> dt.datetime:
    """Parse an RFC 3339 timestamp, guaranteeing the result is timezone-aware."""
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        msg = f"could not parse {field_name} timestamp {value!r}"
        raise ArxivParseError(msg, context={"field": field_name, "value": value}) from exc

    # arXiv always sends an offset, but a naive value here would break every
    # later comparison against a timezone-aware checkpoint.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed


def split_versioned_id(abs_url: str) -> tuple[str, int]:
    """Split an arXiv abstract URL into its identifier and version.

    Returns version 1 when the URL carries no ``vN`` suffix. The API always
    includes one, but identifiers pasted by a human frequently do not.

    Raises:
        ArxivParseError: The URL is not an arXiv abstract URL.

    """
    match = _ABS_URL_PATTERN.match(abs_url.strip())
    if match is None:
        msg = f"not an arXiv abstract URL: {abs_url!r}"
        raise ArxivParseError(msg, context={"url": abs_url})

    version_text = match.group("version")
    return match.group("arxiv_id"), int(version_text) if version_text else 1


def _extract_links(entry: Element) -> tuple[str | None, str | None]:
    """Return ``(abs_url, pdf_url)`` from an entry's link elements."""
    abs_url: str | None = None
    pdf_url: str | None = None

    for link in entry.findall("atom:link", NAMESPACES):
        href = link.get("href")
        if not href:
            continue
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = href
        elif link.get("rel") == "alternate":
            abs_url = href

    return abs_url, pdf_url


def parse_entry(entry: Element) -> ArxivPaper:
    """Convert a single ``<entry>`` element into an :class:`ArxivPaper`.

    Raises:
        ArxivParseError: A required field is missing or malformed.

    """
    raw_id = _text_of(entry.find("atom:id", NAMESPACES))
    if raw_id is None:
        msg = "entry has no <id> element"
        raise ArxivParseError(msg)

    if _ERROR_ID_MARKER in raw_id:
        detail = _text_of(entry.find("atom:summary", NAMESPACES)) or "unspecified"
        msg = f"arXiv returned an error entry: {detail}"
        raise ArxivParseError(msg, context={"id": raw_id})

    arxiv_id, version = split_versioned_id(raw_id)

    title = _text_of(entry.find("atom:title", NAMESPACES))
    abstract = _text_of(entry.find("atom:summary", NAMESPACES))
    if title is None or abstract is None:
        msg = f"entry {arxiv_id} is missing a title or summary"
        raise ArxivParseError(msg, context={"arxiv_id": arxiv_id})

    authors = [
        name
        for author in entry.findall("atom:author", NAMESPACES)
        if (name := _text_of(author.find("atom:name", NAMESPACES))) is not None
    ]
    if not authors:
        msg = f"entry {arxiv_id} lists no authors"
        raise ArxivParseError(msg, context={"arxiv_id": arxiv_id})

    published_raw = _text_of(entry.find("atom:published", NAMESPACES))
    updated_raw = _text_of(entry.find("atom:updated", NAMESPACES))
    if published_raw is None or updated_raw is None:
        msg = f"entry {arxiv_id} is missing published or updated"
        raise ArxivParseError(msg, context={"arxiv_id": arxiv_id})

    primary_element = entry.find("arxiv:primary_category", NAMESPACES)
    primary_category = primary_element.get("term") if primary_element is not None else None

    categories = [
        term
        for category in entry.findall("atom:category", NAMESPACES)
        if (term := category.get("term")) is not None
    ]
    if primary_category is None:
        # Fall back to the first listed category rather than failing: some older
        # entries omit the arxiv:primary_category element entirely.
        if not categories:
            msg = f"entry {arxiv_id} has no categories"
            raise ArxivParseError(msg, context={"arxiv_id": arxiv_id})
        primary_category = categories[0]

    # Primary first, so the ordering carries meaning downstream.
    ordered_categories = [primary_category, *(c for c in categories if c != primary_category)]

    abs_url, pdf_url = _extract_links(entry)
    abs_url = abs_url or raw_id
    # arXiv occasionally omits the PDF link; the URL is derivable from the id.
    pdf_url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}v{version}"

    try:
        return ArxivPaper(
            arxiv_id=arxiv_id,
            version=version,
            title=title,
            abstract=abstract,
            authors=authors,
            published_at=_parse_timestamp(published_raw, field_name="published"),
            updated_at=_parse_timestamp(updated_raw, field_name="updated"),
            primary_category=primary_category,
            categories=ordered_categories,
            doi=_text_of(entry.find("arxiv:doi", NAMESPACES)),
            journal_ref=_text_of(entry.find("arxiv:journal_ref", NAMESPACES)),
            comment=_text_of(entry.find("arxiv:comment", NAMESPACES)),
            abs_url=abs_url,
            pdf_url=pdf_url,
        )
    except ValueError as exc:  # pydantic ValidationError subclasses ValueError
        msg = f"entry {arxiv_id} failed validation: {exc}"
        raise ArxivParseError(msg, context={"arxiv_id": arxiv_id}) from exc


def _int_text(root: Element, path: str, *, default: int = 0) -> int:
    """Read an integer-valued element, tolerating absence and malformed values."""
    text = _text_of(root.find(path, NAMESPACES))
    if text is None:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def parse_feed(xml: str | bytes, *, strict: bool = False) -> ArxivPage:
    """Parse a complete Atom feed into an :class:`ArxivPage`.

    Args:
        xml: The raw response body.
        strict: When True, one unparseable entry fails the whole page. When
            False (the default), bad entries are skipped so a single malformed
            record cannot block an otherwise good batch — which matters for a
            nightly job that should make progress rather than halt.

    Raises:
        ArxivParseError: The document is not well-formed XML, or ``strict`` is
            set and an entry failed.

    """
    try:
        root = DefusedElementTree.fromstring(xml)
    except Exception as exc:  # defusedxml raises several unrelated types
        msg = f"response is not well-formed XML: {exc}"
        raise ArxivParseError(msg) from exc

    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", NAMESPACES):
        try:
            papers.append(parse_entry(entry))
        except ArxivParseError:
            if strict:
                raise
            # Skipped entries are counted by the caller via the difference
            # between itemsPerPage and len(papers); logging happens there, where
            # the run context is available.
            continue

    return ArxivPage(
        papers=papers,
        total_results=_int_text(root, "opensearch:totalResults"),
        start_index=_int_text(root, "opensearch:startIndex"),
        items_per_page=_int_text(root, "opensearch:itemsPerPage", default=len(papers)),
    )
