"""Split a paper into retrievable passages that respect its structure.

Chunking sets the ceiling on retrieval quality, and the naive approach --
cutting every N characters -- fails in a specific, visible way. It produces
passages like::

    ...recall improved by 12% when the reranker was applied to the top-50
    candidate pool. However, this came at a cost of 340ms addi

That fragment begins mid-sentence, ends mid-word, and carries no idea it came
from the Results section. It embeds poorly because it is not about one thing,
and if retrieved it cannot be quoted in an answer.

The strategy here works down a hierarchy, dropping to the next level only when
the current one does not fit:

1. **Sections** are never merged. A chunk spanning the end of Method and the
   start of Results is about neither.
2. **Paragraphs** are the unit of packing. Several small ones may share a chunk;
   a paragraph is never split unless it alone exceeds the limit.
3. **Sentences** split an oversized paragraph, so a cut still lands on a
   grammatical boundary.
4. **Words** split a single oversized sentence -- rare, and usually a table or
   equation that extracted as one long line.

Overlap carries the tail of each chunk into the next, so a passage that happens
to straddle a boundary is still findable from either side.

Limitations, stated plainly:

* Section detection is heuristic (see :mod:`sections`) and is sometimes wrong.
  The chunker never depends on it: an unrecognised structure still chunks
  correctly on paragraphs, just with less useful metadata.
* Token counts are estimated by default rather than exact, so a chunk may land
  slightly over or under target.
* Tables and equations do not survive PDF extraction as coherent text, and no
  chunking strategy repairs that.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Final

from paper_curator.core.config import ChunkingSettings, Settings, get_settings
from paper_curator.core.logging import get_logger
from paper_curator.ingestion.chunking.models import TextChunk
from paper_curator.ingestion.chunking.sections import (
    Section,
    SectionKind,
    split_into_sections,
)
from paper_curator.ingestion.chunking.tokens import TokenCounter, default_token_counter

logger = get_logger(__name__)

# A fragment below this, sitting alongside other content in the same section, is
# discarded: at this size it is a stray heading, a page number or an orphaned
# caption. Distinct from the configurable min_tokens, which is a packing target
# rather than a deletion threshold.
NOISE_FLOOR_TOKENS: Final = 12

# Nothing below this is ever kept, even when it is all a section contains. Four
# tokens cannot carry a retrievable idea.
ABSOLUTE_MIN_TOKENS: Final = 4

# A blank line separates paragraphs in extracted text.
_PARAGRAPH_SPLIT: Final = re.compile(r"\n\s*\n+")

# Sentence end: terminal punctuation, then whitespace, then a capital or digit.
# Requiring the capital avoids splitting on "et al." or "Fig. 3" mid-sentence.
_SENTENCE_SPLIT: Final = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Abbreviations that end in a full stop and are followed by a capital, which the
# pattern above would otherwise treat as a sentence end.
_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "al.",
        "e.g.",
        "i.e.",
        "cf.",
        "vs.",
        "etc.",
        "Fig.",
        "Eq.",
        "Ref.",
        "Sec.",
        "Tab.",
        "App.",
        "Dr.",
        "Prof.",
        "Mr.",
        "Ms.",
        "No.",
        "approx.",
    }
)


def split_paragraphs(text: str) -> list[str]:
    """Split a section's text into paragraphs, discarding blank ones."""
    return [para.strip() for para in _PARAGRAPH_SPLIT.split(text) if para.strip()]


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, keeping common abbreviations intact."""
    pieces = _SENTENCE_SPLIT.split(text.strip())

    merged: list[str] = []
    for piece in pieces:
        # If the previous fragment ended in an abbreviation, the "sentence end"
        # was spurious and the two belong together.
        if merged and any(merged[-1].endswith(abbr) for abbr in _ABBREVIATIONS):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)

    return [sentence for sentence in merged if sentence.strip()]


class StructureAwareChunker:
    """Splits extracted paper text into chunks that respect section structure."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._config: ChunkingSettings = (settings or get_settings()).chunking
        self._count = (token_counter or default_token_counter()).count

    # ------------------------------------------------------------------
    # Splitting text into units small enough to pack
    # ------------------------------------------------------------------

    def _split_oversized_sentence(self, sentence: str) -> list[str]:
        """Split on word boundaries as a last resort.

        Reached only by a "sentence" that alone exceeds the chunk limit, which
        in practice means a table, an equation block, or a reference list that
        extracted as one unbroken line. There is no meaningful boundary left, so
        the cut is arbitrary -- but a truncated chunk would be worse.
        """
        words = sentence.split()
        pieces: list[str] = []
        current: list[str] = []

        for word in words:
            candidate = " ".join([*current, word])
            if current and self._count(candidate) > self._config.max_tokens:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)

        if current:
            pieces.append(" ".join(current))
        return pieces

    def _units_for(self, paragraph: str) -> list[str]:
        """Break a paragraph into pieces that each fit inside one chunk."""
        if self._count(paragraph) <= self._config.max_tokens:
            return [paragraph]

        units: list[str] = []
        for sentence in split_sentences(paragraph):
            if self._count(sentence) <= self._config.max_tokens:
                units.append(sentence)
            else:
                units.extend(self._split_oversized_sentence(sentence))
        return units

    # ------------------------------------------------------------------
    # Packing units into chunks
    # ------------------------------------------------------------------

    def _overlap_units(self, units: Sequence[str]) -> list[str]:
        """Return trailing units totalling about the configured overlap.

        Taken from the end of the chunk just emitted and used to seed the next
        one, so a passage split across a boundary remains findable from either
        side.
        """
        budget = self._config.effective_overlap
        if budget <= 0:
            return []

        carried: list[str] = []
        total = 0
        for unit in reversed(units):
            cost = self._count(unit)
            if total + cost > budget and carried:
                break
            carried.insert(0, unit)
            total += cost
        return carried

    def _pack_section(self, section: Section, start_index: int) -> list[TextChunk]:
        """Turn one section into chunks, never crossing into another section."""
        units: list[str] = []
        for paragraph in split_paragraphs(section.text):
            units.extend(self._units_for(paragraph))

        if not units:
            return []

        is_subsection = section.level > 1
        chunks: list[TextChunk] = []
        pending: list[str] = []
        index = start_index

        def emit() -> None:
            nonlocal pending, index
            body = "\n\n".join(pending).strip()
            if not body:
                return
            chunks.append(
                TextChunk(
                    text=body,
                    chunk_index=index,
                    token_count=self._count(body),
                    section=section.parent_heading if is_subsection else section.heading,
                    subsection=section.heading if is_subsection else None,
                    kind=section.kind,
                )
            )
            index += 1

        for unit in units:
            candidate = [*pending, unit]
            if pending and self._count("\n\n".join(candidate)) > self._config.max_tokens:
                emit()
                # Seed the next chunk with overlap, but only as much as still
                # leaves room for the unit itself. Without this clamp a unit
                # close to the limit plus a full overlap exceeds max_tokens, and
                # the embedding model would silently truncate the result.
                carried = self._overlap_units(pending)
                while (
                    carried and self._count("\n\n".join([*carried, unit])) > self._config.max_tokens
                ):
                    carried.pop(0)
                pending = [*carried, unit]
            else:
                pending = candidate

        emit()
        return chunks

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def _keep(self, section: Section) -> bool:
        """Decide whether a section belongs in the index."""
        if section.kind is SectionKind.REFERENCES:
            return self._config.include_references
        if section.kind is SectionKind.APPENDIX:
            return self._config.include_appendix
        # Acknowledgements are funding statements and thanks: they match author
        # and institution queries for entirely the wrong reason.
        return section.kind is not SectionKind.ACKNOWLEDGEMENTS

    def chunk(self, text: str) -> list[TextChunk]:
        """Split a paper's extracted text into retrievable chunks."""
        if not text.strip():
            return []

        sections = split_into_sections(text)
        kept = [section for section in sections if self._keep(section)]

        chunks: list[TextChunk] = []
        for section in kept:
            chunks.extend(self._pack_section(section, start_index=len(chunks)))
            if len(chunks) >= self._config.max_chunks_per_paper:
                logger.warning(
                    "chunk_limit_reached",
                    limit=self._config.max_chunks_per_paper,
                    detail="remaining sections were not chunked",
                )
                chunks = chunks[: self._config.max_chunks_per_paper]
                break

        final = self._consolidate(chunks)

        logger.info(
            "paper_chunked",
            sections_found=len(sections),
            sections_kept=len(kept),
            chunks=len(final),
            consolidated=len(chunks) - len(final),
        )
        return final

    def _consolidate(self, chunks: Iterable[TextChunk]) -> list[TextChunk]:
        """Merge undersized chunks where possible, drop only genuine noise.

        ``min_tokens`` is a target, not a licence to delete. An early version
        simply dropped every chunk below it, and a short-but-real subsection --
        two paragraphs of Method -- disappeared from the index entirely. Content
        vanishing silently is a far worse failure than a slightly small chunk.

        So an undersized chunk is first merged into the previous chunk *from the
        same section*, if the result still fits. Merging across a section
        boundary is refused, for the same reason sections are never packed
        together: the result would be about neither.

        Only what cannot be merged and falls under :data:`NOISE_FLOOR_TOKENS` is
        discarded. At that size it is a stray heading, a page number, or a
        caption orphaned by extraction -- text that matches noisily and crowds
        out real passages.

        Renumbering afterwards keeps ``chunk_index`` contiguous, which matters
        because it forms part of the deterministic OpenSearch document ID.
        """
        # Materialised up front: the fallback below re-reads this, and an
        # exhausted iterator would make it look as though there was no input.
        candidates = list(chunks)
        # How many chunks each section produced, so a small chunk can be told
        # apart from a small section.
        section_sizes = Counter((c.section, c.subsection) for c in candidates)
        merged: list[TextChunk] = []

        for chunk in candidates:
            if chunk.token_count >= self._config.min_tokens:
                merged.append(chunk)
                continue

            previous = merged[-1] if merged else None
            same_section = (
                previous is not None
                and previous.section == chunk.section
                and previous.subsection == chunk.subsection
            )
            if previous is not None and same_section:
                combined = f"{previous.text}\n\n{chunk.text}"
                if self._count(combined) <= self._config.max_tokens:
                    merged[-1] = TextChunk(
                        text=combined,
                        chunk_index=previous.chunk_index,
                        token_count=self._count(combined),
                        section=previous.section,
                        subsection=previous.subsection,
                        kind=previous.kind,
                    )
                    continue

            # Merging was impossible -- either nothing precedes it, or the
            # previous chunk belongs to a different section, or combining would
            # overflow. Now decide whether it is noise or content.
            #
            # The distinction that matters: a small chunk *alongside* other
            # chunks of the same section is a fragment, and dropping it loses
            # nothing the section does not already say. A small chunk that is
            # the section's entire content is the section, and dropping it
            # removes that part of the paper from the index altogether.
            is_whole_section = section_sizes[(chunk.section, chunk.subsection)] == 1
            if chunk.token_count < ABSOLUTE_MIN_TOKENS:
                continue
            if is_whole_section or chunk.token_count >= NOISE_FLOOR_TOKENS:
                merged.append(chunk)

        # The noise floor exists to drop fragments that sit *alongside* real
        # content. When a paper is short enough that everything falls below it,
        # discarding the lot would remove the paper from the index entirely --
        # so the largest surviving candidate is kept instead. A very short paper
        # is still a paper.
        if not merged and candidates:
            best = max(candidates, key=lambda c: c.token_count)
            logger.info(
                "short_paper_kept",
                token_count=best.token_count,
                noise_floor=NOISE_FLOOR_TOKENS,
                detail="all chunks were below the noise floor; keeping the largest",
            )
            merged = [best]

        return [
            TextChunk(
                text=chunk.text,
                chunk_index=position,
                token_count=chunk.token_count,
                section=chunk.section,
                subsection=chunk.subsection,
                kind=chunk.kind,
            )
            for position, chunk in enumerate(merged)
        ]
