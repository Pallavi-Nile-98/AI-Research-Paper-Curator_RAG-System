"""Structure-aware chunking of academic papers.

    from paper_curator.ingestion.chunking import StructureAwareChunker

    chunks = StructureAwareChunker().chunk(extracted_text)

Chunking on the paper's own structure -- sections, then paragraphs, then
sentences -- rather than every N characters, so each chunk is a coherent passage
that knows where it came from. See :mod:`chunker` for the strategy and its
limitations, and :mod:`sections` for why recovering structure from a PDF is
inherently uncertain.
"""

from paper_curator.ingestion.chunking.chunker import (
    StructureAwareChunker,
    split_paragraphs,
    split_sentences,
)
from paper_curator.ingestion.chunking.models import TextChunk, build_chunk_document
from paper_curator.ingestion.chunking.sections import (
    Heading,
    Section,
    SectionKind,
    detect_heading,
    split_into_sections,
)
from paper_curator.ingestion.chunking.tokens import (
    HeuristicTokenCounter,
    HuggingFaceTokenCounter,
    TokenCounter,
    default_token_counter,
)

__all__ = [
    "Heading",
    "HeuristicTokenCounter",
    "HuggingFaceTokenCounter",
    "Section",
    "SectionKind",
    "StructureAwareChunker",
    "TextChunk",
    "TokenCounter",
    "build_chunk_document",
    "default_token_counter",
    "detect_heading",
    "split_into_sections",
    "split_paragraphs",
    "split_sentences",
]
