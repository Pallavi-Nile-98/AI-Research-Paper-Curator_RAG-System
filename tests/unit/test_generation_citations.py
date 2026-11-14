"""Tests for citation extraction, classification and the prompt contract.

This is the grounding check, so these tests are the evidence behind the claim
that answers are cited rather than merely asked to be. Two properties matter
most and are tested hardest:

* a label the model invented must be **detected**, not quietly accepted
* a label the model cited correctly must not be **mistaken** for invented

The second is as important as the first. A validator that flags real citations
would train anyone reading the logs to ignore it.
"""

from __future__ import annotations

import pytest

from paper_curator.generation.citations import classify, extract_labels, strip_invented
from paper_curator.generation.prompts import (
    PROMPT_VERSION,
    REFUSAL_TOKEN,
    build_messages,
    is_refusal,
)
from paper_curator.retrieval.context import AssembledContext, Citation

pytestmark = pytest.mark.unit


def make_citation(label: str, arxiv_id: str = "2401.00001") -> Citation:
    """Build a Citation with only the fields these tests read."""
    return Citation(
        label=label,
        arxiv_id=arxiv_id,
        version=1,
        title=f"Paper {label}",
        authors=["Doe"],
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        published_at="2024-01-15",
        chunk_ids=[f"{arxiv_id}:0"],
    )


SUPPLIED = [make_citation("P1", "2401.00001"), make_citation("P2", "2401.00002")]


class TestExtractLabels:
    def test_single(self) -> None:
        labels, malformed = extract_labels("A claim [P1].")
        assert labels == ["P1"]
        assert malformed == []

    def test_adjacent_brackets(self) -> None:
        labels, _ = extract_labels("Supported by two sources [P1][P2].")
        assert labels == ["P1", "P2"]

    def test_shared_bracket_comma_separated(self) -> None:
        labels, _ = extract_labels("Both agree [P1, P2].")
        assert labels == ["P1", "P2"]

    def test_shared_bracket_with_and(self) -> None:
        """Scanning for labels handles separators without enumerating them."""
        labels, _ = extract_labels("Both agree [P1 and P3].")
        assert labels == ["P1", "P3"]

    def test_order_of_first_appearance(self) -> None:
        labels, _ = extract_labels("[P3] then [P1] then [P3] again.")
        assert labels == ["P3", "P1"]

    def test_deduplicates(self) -> None:
        labels, _ = extract_labels("[P1] and again [P1].")
        assert labels == ["P1"]

    def test_lowercase_is_accepted(self) -> None:
        """Case variation is a formatting slip, not a grounding failure."""
        labels, malformed = extract_labels("A claim [p1].")
        assert labels == ["P1"]
        assert malformed == []

    def test_space_between_p_and_number(self) -> None:
        labels, _ = extract_labels("A claim [P 2].")
        assert labels == ["P2"]

    def test_double_digit_labels(self) -> None:
        labels, _ = extract_labels("A claim [P10].")
        assert labels == ["P10"]

    def test_leading_zero_normalised(self) -> None:
        labels, _ = extract_labels("A claim [P01].")
        assert labels == ["P1"]

    def test_no_citations(self) -> None:
        labels, malformed = extract_labels("An answer with no sources at all.")
        assert labels == []
        assert malformed == []

    @pytest.mark.parametrize("raw", ["[1]", "[Paper 2]", "[source 3]", "[ref 4]"])
    def test_malformed_shapes_are_recorded_not_dropped(self, raw: str) -> None:
        """A right source in the wrong format must not read as 'cited nothing'.

        Reporting these as zero citations would send someone hunting for a
        retrieval bug when the actual fix is in the prompt.
        """
        labels, malformed = extract_labels(f"A claim {raw}.")
        assert labels == []
        assert len(malformed) == 1

    def test_markdown_link_is_not_a_citation(self) -> None:
        labels, malformed = extract_labels("See [the paper](https://arxiv.org/abs/x) here.")
        assert labels == []
        assert malformed == []

    def test_ordinary_bracketed_prose_is_ignored(self) -> None:
        labels, malformed = extract_labels("The result [see above] was unexpected.")
        assert labels == []
        assert malformed == []

    def test_unclosed_bracket_does_not_swallow_the_text(self) -> None:
        labels, _ = extract_labels("An opening [ bracket that never closes [P1].")
        assert labels == ["P1"]


class TestClassify:
    def test_all_valid(self) -> None:
        used, invented, malformed = classify("Claim [P1] and claim [P2].", SUPPLIED)
        assert [c.label for c in used] == ["P1", "P2"]
        assert invented == []
        assert malformed == []

    def test_detects_invented_label(self) -> None:
        """The failure the whole module exists to catch."""
        used, invented, _ = classify("A fabricated attribution [P9].", SUPPLIED)
        assert used == []
        assert invented == ["P9"]

    def test_mixed_real_and_invented(self) -> None:
        used, invented, _ = classify("Real [P1], invented [P7].", SUPPLIED)
        assert [c.label for c in used] == ["P1"]
        assert invented == ["P7"]

    def test_used_follows_supplied_order_not_prose_order(self) -> None:
        """So a rendered source list reads P1, P2 regardless of the prose."""
        used, _, _ = classify("First [P2], then [P1].", SUPPLIED)
        assert [c.label for c in used] == ["P1", "P2"]

    def test_uncited_source_is_not_an_error(self) -> None:
        used, invented, malformed = classify("Only one source [P1].", SUPPLIED)
        assert [c.label for c in used] == ["P1"]
        assert invented == []
        assert malformed == []

    def test_no_citations_at_all(self) -> None:
        used, invented, malformed = classify("An unattributed answer.", SUPPLIED)
        assert used == []
        assert invented == []
        assert malformed == []

    def test_empty_supplied_makes_every_label_invented(self) -> None:
        used, invented, _ = classify("A claim [P1].", [])
        assert used == []
        assert invented == ["P1"]


class TestStripInvented:
    def test_removes_standalone_label(self) -> None:
        assert "[P9]" not in strip_invented("A claim [P9].", ["P9"])

    def test_keeps_valid_labels(self) -> None:
        cleaned = strip_invented("Real [P1], invented [P9].", ["P9"])
        assert "[P1]" in cleaned
        assert "[P9]" not in cleaned

    def test_removes_from_shared_bracket(self) -> None:
        cleaned = strip_invented("Both [P1, P9].", ["P9"])
        assert "P9" not in cleaned
        assert "P1" in cleaned

    def test_no_op_when_nothing_invented(self) -> None:
        text = "A clean answer [P1]."
        assert strip_invented(text, []) == text

    def test_preserves_paragraph_structure(self) -> None:
        cleaned = strip_invented("First para [P9].\n\nSecond para [P1].", ["P9"])
        assert "\n\n" in cleaned


class TestBuildMessages:
    def _context(self, citations: list[Citation]) -> AssembledContext:
        return AssembledContext(
            text="[P1] Title (Doe, 2024)\nSome passage text.",
            citations=citations,
            token_count=20,
            chunks_included=len(citations),
        )

    def test_produces_system_and_user_messages(self) -> None:
        messages = build_messages("What is BM25?", self._context(SUPPLIED))
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_question_appears_after_context(self) -> None:
        """Attention skews recent, so the question must come last."""
        messages = build_messages("What is BM25?", self._context(SUPPLIED))
        user = messages[1]["content"]
        assert user.index("Some passage text") < user.index("What is BM25?")

    def test_system_prompt_shows_the_citation_format(self) -> None:
        """A worked example, not a description -- the point of the prompt."""
        system = build_messages("q", self._context(SUPPLIED))[0]["content"]
        assert "[P1]" in system

    def test_system_prompt_states_the_refusal_token(self) -> None:
        system = build_messages("q", self._context(SUPPLIED))[0]["content"]
        assert REFUSAL_TOKEN in system

    def test_question_is_stripped(self) -> None:
        messages = build_messages("  What is BM25?  ", self._context(SUPPLIED))
        assert "What is BM25?\n" in messages[1]["content"]

    def test_empty_context_is_refused(self) -> None:
        """Generating from no evidence is the failure this system prevents."""
        empty = AssembledContext(text="", citations=[], token_count=0, chunks_included=0)
        with pytest.raises(ValueError, match="empty context"):
            build_messages("What is BM25?", empty)

    def test_prompt_version_is_set(self) -> None:
        assert PROMPT_VERSION


class TestIsRefusal:
    def test_bare_token(self) -> None:
        assert is_refusal(REFUSAL_TOKEN)

    def test_token_with_surrounding_courtesy(self) -> None:
        """Models emit the token reliably and resist stopping there."""
        assert is_refusal(f"{REFUSAL_TOKEN}\n\nSorry, I could not find this.")

    def test_ordinary_answer_is_not_a_refusal(self) -> None:
        assert not is_refusal("BM25 is a ranking function [P1].")

    def test_hedged_answer_is_not_a_refusal(self) -> None:
        """A hedge is a good answer, and phrase-matching would discard it."""
        assert not is_refusal(
            "It is not entirely clear from [P2] whether this holds, but [P1] suggests it does."
        )

    def test_long_answer_mentioning_the_token_is_not_a_refusal(self) -> None:
        long_answer = f"The sentinel {REFUSAL_TOKEN} is used when " + ("padding " * 60)
        assert not is_refusal(long_answer)
