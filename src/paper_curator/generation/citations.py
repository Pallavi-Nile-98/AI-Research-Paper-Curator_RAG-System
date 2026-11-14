"""Extracting and checking the citations in a generated answer.

This module is the reason the system can call its answers *grounded* rather
than merely *instructed to be grounded*. The prompt tells the model to cite
only the supplied sources; this verifies whether it did.

That distinction is not pedantic. A 3B model given three sources labelled
``[P1]`` to ``[P3]`` will sometimes write ``[P4]``, and the sentence around it
reads exactly like every other sentence. Nothing downstream notices: the answer
is fluent, the citation is formatted correctly, and the label points at
nothing. Prompt engineering lowers the rate and does not reach zero, so the
only way to keep the guarantee honest is to check after the fact and report
what was found.

**Parsing is lenient, classification is strict.** The parser accepts shapes the
format does not define -- ``[1]``, ``[Paper 2]``, ``[p3]`` -- because a model
citing the right source in the wrong format is a different failure from one
inventing a source, and a parser matching only the exact form would report the
first as "cited nothing". Those two failures have different fixes, so they are
counted separately.
"""

from __future__ import annotations

import re

from paper_curator.retrieval.context import Citation

# Bracketed spans, bounded in length so a stray bracket cannot swallow a
# paragraph and so a markdown link's text is examined rather than its URL.
_BRACKET = re.compile(r"\[([^\[\]]{1,60})\]")

# A well-formed label inside a bracket. Case-insensitive and tolerant of one
# space, so "[p 1]" is read as P1: case is not a grounding failure.
_LABEL = re.compile(r"\bP\s?(\d+)\b", re.IGNORECASE)

# Citation-shaped content that the format does not define. A bare number, or a
# spelled-out reference. Matched only when no well-formed label is present.
_MALFORMED = re.compile(
    r"^\s*(?:(?:paper|source|ref|reference|citation)\s*)?(\d+)\s*$", re.IGNORECASE
)


def extract_labels(text: str) -> tuple[list[str], list[str]]:
    """Pull citation labels out of an answer.

    Returns:
        ``(labels, malformed)`` -- normalised ``P<n>`` labels in order of first
        appearance, and the raw text of bracketed spans that looked like
        citation attempts but did not match the format.

    """
    labels: list[str] = []
    malformed: list[str] = []
    seen: set[str] = set()

    for match in _BRACKET.finditer(text):
        content = match.group(1)

        # Several labels can share one bracket: "[P1, P2]", "[P1 and P2]".
        # Scanning for labels rather than splitting on punctuation handles
        # every separator a model might choose without enumerating them.
        found = _LABEL.findall(content)
        if found:
            for number in found:
                label = f"P{int(number)}"
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
            continue

        if _MALFORMED.match(content):
            malformed.append(content.strip())

    return labels, malformed


def classify(text: str, supplied: list[Citation]) -> tuple[list[Citation], list[str], list[str]]:
    """Split an answer's citations into supported, invented and malformed.

    Args:
        text: The generated answer.
        supplied: Every source the model was given.

    Returns:
        ``(used, invented, malformed)`` -- the supplied citations the answer
        references, labels it cited that were never supplied, and
        citation-shaped text in an undefined format.

    """
    labels, malformed = extract_labels(text)
    by_label = {citation.label: citation for citation in supplied}

    invented = [label for label in labels if label not in by_label]

    # Returned in supplied order, not order of appearance, so a rendered
    # source list reads P1, P2, P3 regardless of the order the prose used
    # them.
    cited = {label for label in labels if label in by_label}
    used = [citation for citation in supplied if citation.label in cited]

    return used, invented, malformed


def strip_invented(text: str, invented: list[str]) -> str:
    """Remove invented citation labels from an answer.

    Offered but **not applied by default**, and the distinction is deliberate.
    Stripping produces a cleaner answer while destroying the evidence that the
    model fabricated a source: the sentence remains, now reading as an
    unattributed claim, and the quality signal is gone. The caller decides,
    and :class:`~paper_curator.generation.models.GeneratedAnswer` keeps the
    original either way.
    """
    if not invented:
        return text

    cleaned = text
    for label in invented:
        # Also catch the label inside a shared bracket, e.g. "[P1, P9]".
        cleaned = re.sub(rf"\[\s*{re.escape(label)}\s*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            rf"(?<=\[)\s*{re.escape(label)}\s*[,;]\s*|\s*[,;]\s*{re.escape(label)}\s*(?=\])",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    # Collapse the double spaces a removal leaves behind, without touching
    # newlines, so paragraph structure survives.
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()
