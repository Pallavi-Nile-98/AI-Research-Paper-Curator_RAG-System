"""What a generated answer is made of.

The central type is :class:`GeneratedAnswer`, and it carries more than the
text. A RAG system's claim is not "here is an answer" but "here is an answer
supported by these specific sources", and that claim is only checkable if the
evidence travels with it.

So the answer records which citations the model **actually used**, which
sources were supplied but ignored, and -- the field that does the real work --
which labels the model **invented**. A 3B model asked to cite three papers will
sometimes write ``[P7]``. Prompt instructions reduce that; they do not
eliminate it. Measuring it is what turns "grounded" from a claim into a
property that can be reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from paper_curator.retrieval.context import Citation


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Tokens consumed by one generation, as reported by the server."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A raw completion, before any grounding checks.

    Deliberately separate from :class:`GeneratedAnswer`: this is what the
    model said, that is what the system is willing to stand behind. Keeping
    the two apart means the validation step has something to validate rather
    than mutating a value that already claims to be an answer.
    """

    text: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = "stop"
    duration_ms: int = 0

    @property
    def hit_token_limit(self) -> bool:
        """True when output stopped because it ran out of room.

        Worth checking before trusting a long answer: a response cut off at
        the limit can end mid-sentence, and its final citation may be missing
        entirely.
        """
        return self.finish_reason == "length"


class RefusalReason:
    """Why the system declined to answer, when it did.

    Strings rather than an enum because these are reported to users and
    logged; the set is small and stable enough that an enum would add
    ceremony without adding safety.
    """

    NO_EVIDENCE = "no_evidence"
    """Retrieval returned nothing. The model was never called."""

    MODEL_DECLINED = "model_declined"
    """Context existed, but the model judged it insufficient and said so."""


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """An answer, the sources behind it, and what is wrong with it.

    Every field that describes a problem defaults to the benign value, so a
    clean answer reads simply, while a flawed one carries the detail needed to
    diagnose it.
    """

    text: str
    question: str

    citations_used: list[Citation] = field(default_factory=list)
    """Sources the answer actually cites."""

    citations_supplied: list[Citation] = field(default_factory=list)
    """Everything retrieval provided, cited or not."""

    invented_labels: list[str] = field(default_factory=list)
    """Labels the model cited that were never supplied.

    The direct measure of ungrounded attribution. Non-empty means the answer
    points at evidence that does not exist.
    """

    malformed_labels: list[str] = field(default_factory=list)
    """Citation-like strings in a shape the format does not define, e.g. ``[1]``.

    Tracked separately from invented labels because the failures differ. An
    invented label is a fabricated source; a malformed one is usually the right
    source written the wrong way, which is a prompt problem rather than a
    grounding problem.
    """

    refused: bool = False
    refusal_reason: str | None = None

    model: str = ""
    prompt_version: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)

    retrieval_ms: int = 0
    generation_ms: int = 0
    truncated_context: bool = False
    hit_token_limit: bool = False

    @property
    def total_ms(self) -> int:
        return self.retrieval_ms + self.generation_ms

    @property
    def is_grounded(self) -> bool:
        """True when every citation in the answer points at a supplied source.

        Note what this does **not** claim: that the cited paper actually
        supports the sentence it is attached to. Verifying that needs either a
        human or a separate entailment model. This checks that the citation
        refers to something real, which is a necessary condition for grounding
        and not a sufficient one.
        """
        return not self.invented_labels

    @property
    def uncited_sources(self) -> list[Citation]:
        """Supplied sources the answer never referenced.

        Not a defect on its own -- retrieval returns candidates, and a good
        answer may need only some. Consistently high counts mean retrieval is
        returning more than the question needs.
        """
        used = {citation.label for citation in self.citations_used}
        return [citation for citation in self.citations_supplied if citation.label not in used]

    @property
    def has_problems(self) -> bool:
        """True when anything about this answer warrants a second look."""
        return bool(self.invented_labels or self.malformed_labels or self.hit_token_limit)

    def summary(self) -> str:
        """One line for logs, naming problems rather than hiding them."""
        if self.refused:
            return f"refused ({self.refusal_reason})"

        parts = [
            f"{len(self.text)} chars",
            f"{len(self.citations_used)}/{len(self.citations_supplied)} sources cited",
            f"{self.total_ms}ms",
        ]
        if self.invented_labels:
            parts.append(f"INVENTED {self.invented_labels}")
        if self.malformed_labels:
            parts.append(f"malformed {self.malformed_labels}")
        if self.hit_token_limit:
            parts.append("hit token limit")
        return ", ".join(parts)
