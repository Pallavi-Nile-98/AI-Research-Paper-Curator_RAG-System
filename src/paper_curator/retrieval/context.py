"""Assembling retrieved chunks into prompt-ready, citable context.

This is the handover from retrieval to generation, and it decides what the
model is physically able to say. A passage left out cannot be cited; a passage
included without a stable label cannot be attributed.

Three decisions shape it.

**Citation labels are per paper, not per chunk.** Three passages from one paper
all carry ``[P1]``. That matches how citation actually works -- you cite a
paper, not a fragment -- and it keeps the label set small enough for a small
model to use reliably. Labelling every chunk separately gives an eight-passage
context eight labels, and a 3B model starts mixing them up. The section is
still shown on each passage, so the model can be specific about *where* in the
paper without needing a separate identifier for it.

**Labels are assigned by first appearance**, so ``[P1]`` is always the
highest-ranked source. Stable within one answer, which is all that matters:
they are generated per request and never stored.

**The budget is enforced in tokens, and a chunk is never truncated.** Half a
passage is worse than no passage: it can end mid-sentence, mid-claim, or
mid-negation, and a model quoting the visible half will cite something the
paper did not say. Chunks are taken whole until the next one will not fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.logging import get_logger
from paper_curator.ingestion.chunking.tokens import TokenCounter, default_token_counter
from paper_curator.retrieval.models import RetrievedChunk

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Citation:
    """One cited source, and everything needed to render or verify it."""

    label: str
    """Identifier used in the answer, e.g. ``P1``."""

    arxiv_id: str
    version: int
    title: str
    authors: list[str]
    abs_url: str
    pdf_url: str
    published_at: str

    sections: list[str] = field(default_factory=list)
    """Which parts of the paper were supplied, in the order they appeared."""

    chunk_ids: list[str] = field(default_factory=list)
    """Exact chunks behind this citation, for verification and failure analysis."""

    @property
    def author_summary(self) -> str:
        """Short author string for display, e.g. "Doe et al."."""
        if not self.authors:
            return "unknown"
        if len(self.authors) == 1:
            return self.authors[0]
        return f"{self.authors[0]} et al."

    @property
    def year(self) -> str:
        """Publication year, or an empty string when unparseable."""
        return self.published_at[:4] if len(self.published_at) >= 4 else ""


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """Prompt-ready evidence plus the structured sources behind it."""

    text: str
    """The block inserted into the prompt."""

    citations: list[Citation]
    token_count: int
    chunks_included: int
    chunks_dropped: int = 0
    truncated: bool = False
    """True when the budget stopped some retrieved chunks from being included."""

    @property
    def is_empty(self) -> bool:
        """True when retrieval produced nothing to answer from.

        The caller must treat this as insufficient evidence rather than asking
        the model to answer anyway -- an unsupported answer is the exact
        failure this system exists to avoid.
        """
        return not self.citations

    @property
    def valid_labels(self) -> set[str]:
        """Labels the answer is permitted to cite.

        Citation validation compares what the model actually cited against
        this. Anything outside it was invented.
        """
        return {citation.label for citation in self.citations}

    def summary(self) -> str:
        return (
            f"{self.chunks_included} chunks from {len(self.citations)} papers, "
            f"{self.token_count} tokens"
            + (f", {self.chunks_dropped} dropped for budget" if self.chunks_dropped else "")
        )


class ContextBuilder:
    """Turns ranked chunks into a citable context block."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        token_counter: TokenCounter | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._budget = resolved.retrieval.context_token_budget
        self._count = (token_counter or default_token_counter()).count

    @staticmethod
    def _render_passage(label: str, chunk: RetrievedChunk, citation: Citation) -> str:
        """Render one passage with its attribution header.

        The header repeats on every passage rather than grouping by paper.
        Grouping is tidier to read and worse for the model: with a shared
        heading above several passages, a small model attributes the last one
        to whichever label appeared most recently.
        """
        year = f", {citation.year}" if citation.year else ""
        header = f"[{label}] {citation.title} ({citation.author_summary}{year})"
        if chunk.section:
            header += f" — {chunk.location}"
        return f"{header}\n{chunk.text.strip()}"

    def build(self, chunks: list[RetrievedChunk]) -> AssembledContext:
        """Assemble ranked chunks into context, honouring the token budget.

        Chunks are consumed in the order given, which is the ranked order, so
        the budget is spent on the most relevant evidence first.
        """
        if not chunks:
            return AssembledContext(text="", citations=[], token_count=0, chunks_included=0)

        labels: dict[str, str] = {}
        citations: dict[str, Citation] = {}
        passages: list[str] = []
        used_tokens = 0
        dropped = 0

        for chunk in chunks:
            paper_key = chunk.versioned_id

            if paper_key not in labels:
                label = f"P{len(labels) + 1}"
                labels[paper_key] = label
                citations[paper_key] = Citation(
                    label=label,
                    arxiv_id=chunk.arxiv_id,
                    version=chunk.version,
                    title=chunk.title,
                    authors=list(chunk.authors),
                    abs_url=chunk.abs_url,
                    pdf_url=chunk.pdf_url,
                    published_at=chunk.published_at,
                )

            label = labels[paper_key]
            passage = self._render_passage(label, chunk, citations[paper_key])
            cost = self._count(passage)

            # Whole chunks only. A truncated passage can end mid-negation, and
            # a model quoting the visible half cites something the paper did
            # not say.
            if used_tokens + cost > self._budget:
                dropped += 1
                # Keep going rather than stopping: a later chunk may be small
                # enough to fit, and dropping everything after the first
                # oversized passage would waste the remaining budget.
                continue

            passages.append(passage)
            used_tokens += cost

            existing = citations[paper_key]
            citations[paper_key] = Citation(
                label=existing.label,
                arxiv_id=existing.arxiv_id,
                version=existing.version,
                title=existing.title,
                authors=existing.authors,
                abs_url=existing.abs_url,
                pdf_url=existing.pdf_url,
                published_at=existing.published_at,
                sections=[*existing.sections, chunk.location],
                chunk_ids=[*existing.chunk_ids, chunk.chunk_id],
            )

        # A paper whose every passage was dropped for budget must not appear as
        # a citation: the answer would reference evidence the model never saw.
        used_citations = [citation for citation in citations.values() if citation.chunk_ids]
        # Relabel so the surviving sources are P1..Pn with no gaps.
        relabelled: list[Citation] = []
        remap: dict[str, str] = {}
        for index, citation in enumerate(used_citations, start=1):
            new_label = f"P{index}"
            remap[citation.label] = new_label
            relabelled.append(
                Citation(
                    label=new_label,
                    arxiv_id=citation.arxiv_id,
                    version=citation.version,
                    title=citation.title,
                    authors=citation.authors,
                    abs_url=citation.abs_url,
                    pdf_url=citation.pdf_url,
                    published_at=citation.published_at,
                    sections=citation.sections,
                    chunk_ids=citation.chunk_ids,
                )
            )

        text = "\n\n".join(passages)
        for old, new in remap.items():
            if old != new:
                text = text.replace(f"[{old}]", f"[{new}]")

        context = AssembledContext(
            text=text,
            citations=relabelled,
            token_count=used_tokens,
            chunks_included=len(passages),
            chunks_dropped=dropped,
            truncated=dropped > 0,
        )
        logger.debug("context_assembled", summary=context.summary())
        return context
