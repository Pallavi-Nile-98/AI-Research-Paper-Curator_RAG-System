"""Recover a paper's section structure from flat extracted text.

This is the least certain part of the pipeline, and it is worth being honest
about why. **A PDF has no headings.** It has glyphs at coordinates. "3.1
Training Details" is not marked up as a subheading anywhere -- it is text that
happens to sit on its own line in a larger font. Once PyMuPDF extracts it, even
the font is gone and only the line remains.

So structure has to be inferred, and inference is sometimes wrong. The detector
combines signals that rarely coincide by accident:

* **Numbering.** ``3.1 Training Details`` or ``IV. RESULTS``. The strongest
  signal, and it also yields nesting depth for free.
* **Canonical names.** Academic papers reuse a small vocabulary -- Abstract,
  Introduction, Related Work, Method, Results, Discussion, Conclusion,
  References. A short line matching one is almost always a heading.
* **Shape.** Headings are short, start with a capital, and do not end in a full
  stop. Body sentences usually do the opposite.

Known failure modes, none of which are fully solvable here:

* **Two-column layouts.** Extraction sometimes interleaves columns, so a heading
  is spliced into the middle of a body line and never appears alone.
* **Heavy mathematics.** Display equations extract as fragmentary lines that can
  look like numbered headings.
* **Unconventional structures.** Some venues use "Materials and Methods", some
  papers number from zero, some have no numbering at all.

Because detection is unreliable, the chunker never *depends* on it. A paper
whose structure is not recognised still chunks correctly on paragraph
boundaries; it simply carries less useful section metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

MAX_HEADING_CHARS: Final = 90
MIN_HEADING_CHARS: Final = 3


class SectionKind(StrEnum):
    """What role a section plays in the paper.

    Drives decisions the chunker makes: references are usually excluded from the
    index, the abstract is worth keeping whole, and appendices are kept but
    flagged.
    """

    TITLE = "title"
    ABSTRACT = "abstract"
    BODY = "body"
    ACKNOWLEDGEMENTS = "acknowledgements"
    REFERENCES = "references"
    APPENDIX = "appendix"


# Canonical heading text mapped to its role. Matched case-insensitively against
# the heading with any numbering stripped.
_CANONICAL_KINDS: Final[dict[str, SectionKind]] = {
    "abstract": SectionKind.ABSTRACT,
    "summary": SectionKind.ABSTRACT,
    "acknowledgement": SectionKind.ACKNOWLEDGEMENTS,
    "acknowledgements": SectionKind.ACKNOWLEDGEMENTS,
    "acknowledgment": SectionKind.ACKNOWLEDGEMENTS,
    "acknowledgments": SectionKind.ACKNOWLEDGEMENTS,
    "references": SectionKind.REFERENCES,
    "bibliography": SectionKind.REFERENCES,
    "works cited": SectionKind.REFERENCES,
    "appendix": SectionKind.APPENDIX,
    "appendices": SectionKind.APPENDIX,
    "supplementary material": SectionKind.APPENDIX,
    "supplementary materials": SectionKind.APPENDIX,
}

# Recognised as headings even without numbering. Anything not listed still
# matches via the numbering or shape rules.
_CANONICAL_BODY_HEADINGS: Final[frozenset[str]] = frozenset(
    {
        "introduction",
        "background",
        "related work",
        "prior work",
        "motivation",
        "method",
        "methods",
        "methodology",
        "approach",
        "model",
        "architecture",
        "materials and methods",
        "experimental setup",
        "experiments",
        "experimental results",
        "evaluation",
        "results",
        "analysis",
        "ablation study",
        "ablation studies",
        "discussion",
        "limitations",
        "future work",
        "conclusion",
        "conclusions",
        "conclusion and future work",
    }
)

# "3 Methods", "3.1 Training", "3.1.2 Optimiser"
_NUMBERED = re.compile(r"^(?P<number>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<title>\S.*)$")
# "IV. RESULTS" -- common in IEEE-style papers.
_ROMAN = re.compile(r"^(?P<number>[IVXL]{1,6})\.\s+(?P<title>\S.*)$", re.IGNORECASE)
# "Appendix A: Proofs"
_LETTERED_APPENDIX = re.compile(r"^appendix\s+(?P<number>[A-Z])\b[.:]?\s*(?P<title>.*)$", re.I)

_TRAILING_PUNCT = re.compile(r"[.,;:]+$")


@dataclass(frozen=True, slots=True)
class Heading:
    """A line identified as a section heading."""

    text: str
    """Heading text with any numbering removed, e.g. "Training Details"."""

    number: str | None
    """The numbering as written, e.g. "3.1", or None when unnumbered."""

    level: int
    """1 for a section, 2 for a subsection, and so on."""

    kind: SectionKind


@dataclass(frozen=True, slots=True)
class Section:
    """A contiguous region of the paper under one heading."""

    heading: str | None
    """None for text appearing before any recognised heading."""

    kind: SectionKind
    text: str
    level: int = 1
    parent_heading: str | None = None
    """The enclosing section's heading when this is a subsection."""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def _normalise(value: str) -> str:
    """Lowercase, strip numbering-adjacent punctuation and collapse whitespace."""
    cleaned = _TRAILING_PUNCT.sub("", value.strip())
    return " ".join(cleaned.lower().split())


def _present(title: str) -> str:
    """Normalise a heading for display.

    Section names are shown in citations, so an ALL-CAPS heading is title-cased.
    Otherwise the same section would read as "EXPERIMENTAL SETUP" from an
    IEEE-style paper and "Experimental Setup" from an ACL one, and the
    inconsistency would surface in the user interface.
    """
    cleaned = _TRAILING_PUNCT.sub("", title.strip())
    letters = [c for c in cleaned if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return cleaned.title()
    return cleaned


def _classify(title: str) -> SectionKind:
    """Map heading text to a section role."""
    normalised = _normalise(title)
    if normalised in _CANONICAL_KINDS:
        return _CANONICAL_KINDS[normalised]
    # "Appendix A", "Appendix B: Proofs" -- prefix match rather than exact.
    for prefix, kind in _CANONICAL_KINDS.items():
        if normalised.startswith(f"{prefix} ") or normalised.startswith(f"{prefix}:"):
            return kind
    return SectionKind.BODY


def _looks_like_prose(line: str) -> bool:
    """Report whether a line reads as a sentence rather than a heading.

    Guards the shape-based rules against false positives. Without it, a body
    sentence beginning "2. We then applied..." would be taken as a heading.
    """
    stripped = line.strip()
    if stripped.endswith((".", "?", "!", ",", ";", ":")):
        return True
    # A heading is a label, not a clause. Several sentence-like words in a short
    # line indicate prose.
    return len(stripped.split()) > 12


def detect_heading(line: str) -> Heading | None:
    """Identify ``line`` as a section heading, or return None.

    Deliberately conservative. A missed heading costs some section metadata; a
    false positive splits a paragraph in half and produces two incoherent
    chunks, which is the worse outcome.
    """
    stripped = line.strip()
    if not (MIN_HEADING_CHARS <= len(stripped) <= MAX_HEADING_CHARS):
        return None

    # --- Appendix with a letter, checked first: "Appendix A" also matches the
    # canonical rule but the letter carries useful numbering. ---------------
    if (match := _LETTERED_APPENDIX.match(stripped)) is not None:
        title = match.group("title").strip() or "Appendix"
        return Heading(
            text=f"Appendix {match.group('number')}"
            + (f": {title}" if title != "Appendix" else ""),
            number=match.group("number"),
            level=1,
            kind=SectionKind.APPENDIX,
        )

    # --- Numbered ----------------------------------------------------------
    if (match := _NUMBERED.match(stripped)) is not None:
        title = match.group("title").strip()
        if not _looks_like_prose(title) and title[:1].isupper():
            number = match.group("number")
            return Heading(
                text=_present(title),
                number=number,
                level=number.count(".") + 1,
                kind=_classify(title),
            )

    # --- Roman numerals ----------------------------------------------------
    if (match := _ROMAN.match(stripped)) is not None:
        title = match.group("title").strip()
        if not _looks_like_prose(title) and title[:1].isupper():
            return Heading(
                text=_present(title),
                number=match.group("number").upper(),
                level=1,
                kind=_classify(title),
            )

    # --- Canonical names, unnumbered ---------------------------------------
    normalised = _normalise(stripped)
    if normalised in _CANONICAL_KINDS or normalised in _CANONICAL_BODY_HEADINGS:
        return Heading(
            text=_present(stripped),
            number=None,
            level=1,
            kind=_classify(stripped),
        )

    # --- Shape: a short ALL-CAPS line --------------------------------------
    letters = [c for c in stripped if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and 1 < len(stripped.split()) <= 8:
        return Heading(
            text=_present(stripped),
            number=None,
            level=1,
            kind=_classify(stripped),
        )

    return None


def split_into_sections(text: str) -> list[Section]:
    """Split extracted text into sections at detected headings.

    Text appearing before the first heading becomes a section with
    ``heading=None``. In a typical paper that is the title block and author
    list, which is worth keeping rather than discarding.

    A paper where no heading is recognised returns a single unheaded section,
    and the chunker handles that case normally.
    """
    # Page separators inserted by extraction are boundaries, not content.
    lines = text.replace("\f", "\n").split("\n")

    sections: list[Section] = []
    current_heading: Heading | None = None
    current_parent: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if not body and current_heading is None:
            return
        sections.append(
            Section(
                heading=current_heading.text if current_heading else None,
                kind=current_heading.kind if current_heading else SectionKind.TITLE,
                text=body,
                level=current_heading.level if current_heading else 1,
                parent_heading=current_parent,
            )
        )

    for line in lines:
        heading = detect_heading(line)
        if heading is None:
            buffer.append(line)
            continue

        flush()
        buffer = []

        # Track the enclosing section so a subsection knows its parent. A
        # level-1 heading resets it; deeper headings inherit the last level-1.
        if heading.level == 1:
            current_parent = None
        elif current_heading is not None and current_heading.level == 1:
            current_parent = current_heading.text
        current_heading = heading

    flush()

    return [section for section in sections if not section.is_empty or section.heading]
