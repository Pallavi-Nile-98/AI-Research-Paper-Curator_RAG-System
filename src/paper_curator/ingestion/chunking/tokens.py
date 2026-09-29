"""Token counting for chunk sizing.

Chunk limits are expressed in tokens because that is the unit the embedding
model actually consumes: exceed its input limit and the tail of a chunk is
silently dropped, so text would be indexed while contributing nothing to its own
embedding.

Counting tokens exactly requires the model's own tokenizer, which lives behind
the optional ``[ml]`` extra and drags in PyTorch. Chunking would then be
untestable without a 2 GB dependency, and re-chunking experiments would need a
model loaded just to measure text. So counting sits behind a small protocol with
two implementations:

* :class:`HeuristicTokenCounter` -- the default. No dependencies, fast,
  deterministic, and close enough for sizing decisions.
* :class:`HuggingFaceTokenCounter` -- wraps a real tokenizer when one is
  available, for exactness where it matters.

The heuristic is calibrated for English academic prose under a WordPiece
tokenizer, which is what the default embedding model uses. It is an estimate and
is documented as one; the consequence of a small error is a chunk slightly over
or under target, not a broken pipeline.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

# WordPiece splits unfamiliar or long words into several subword tokens, so
# token count exceeds word count. Around 1.3 is the usual ratio for ordinary
# English; academic text sits higher because of technical vocabulary.
_TOKENS_PER_WORD = 1.35

# Words longer than this are typically split into multiple subwords --
# "backpropagation", "hyperparameter", "tokenization".
_LONG_WORD_CHARS = 9
_EXTRA_TOKENS_PER_LONG_WORD = 0.8

# Punctuation and digits are usually tokens in their own right.
_SYMBOL_PATTERN = re.compile(r"[^\w\s]|\d+")
_WORD_PATTERN = re.compile(r"\b[^\W\d_]+\b")


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens in a string."""

    def count(self, text: str) -> int:
        """Return the number of tokens ``text`` occupies."""
        ...


class HeuristicTokenCounter:
    """Estimates token count without loading a tokenizer.

    Accurate to roughly ten percent on English academic prose, which is well
    inside the tolerance for deciding where to cut a chunk. It deliberately
    errs slightly high, because overestimating produces a chunk a little under
    the model's limit while underestimating produces one that gets truncated.
    """

    def count(self, text: str) -> int:
        if not text.strip():
            return 0

        words = _WORD_PATTERN.findall(text)
        symbols = _SYMBOL_PATTERN.findall(text)
        long_words = sum(1 for word in words if len(word) > _LONG_WORD_CHARS)

        estimate = (
            len(words) * _TOKENS_PER_WORD + long_words * _EXTRA_TOKENS_PER_LONG_WORD + len(symbols)
        )
        return max(1, round(estimate))


class HuggingFaceTokenCounter:
    """Counts tokens with a real tokenizer.

    Used where exactness matters and the ``[ml]`` extra is installed. The
    tokenizer is injected rather than loaded here, so this module still imports
    without transformers present.
    """

    def __init__(self, tokenizer: object) -> None:
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            msg = "tokenizer must expose a callable encode()"
            raise TypeError(msg)
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        # add_special_tokens=False: [CLS] and [SEP] are added per model input,
        # not per fragment, so including them would overcount when measuring a
        # piece of a chunk.
        encoded = self._tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        return len(encoded)


def default_token_counter() -> TokenCounter:
    """Return the counter used when the caller supplies none."""
    return HeuristicTokenCounter()
