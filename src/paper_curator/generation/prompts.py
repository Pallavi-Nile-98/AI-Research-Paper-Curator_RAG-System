"""Version-controlled prompt templates.

The prompt is a dependency, not a string literal. Changing it changes answer
quality as surely as changing the retrieval model does, and a quality
regression that cannot be attributed to a specific prompt version cannot be
diagnosed. So :data:`PROMPT_VERSION` is bumped whenever the text changes and is
recorded on every generated answer.

Four choices in the text below are deliberate and worth stating, because each
exists to prevent a specific observed failure.

**The rules are in the system message; the evidence is in the user message.**
Instructions placed next to the evidence compete with it for attention, and a
small model starts treating the rules as more context to summarise.

**The question appears after the context.** Attention in a decoder-only model
skews toward recent tokens, so the last thing before the answer should be what
the answer is about. With the question first, a 3B model asked something narrow
about a 3,000-token context tends to summarise the context instead.

**Refusal is a specific token, not a phrasing.** The model is told to emit
``INSUFFICIENT_EVIDENCE`` exactly. Detecting refusal by searching for apologetic
phrasing is unreliable in both directions: it misses "The passages above do not
address this", and it fires on "It is not clear from [P2] whether ...", which is
a perfectly good hedged answer. A sentinel is checkable.

**Citation format is shown, not described.** "Cite your sources using square
brackets" produces ``[Smith et al. 2024]``, ``[1]`` and ``[Paper 1]`` in roughly
equal measure. A worked example produces ``[P1]``.

None of this is sufficient on its own. Every instruction here is a request the
model can ignore, which is why
:mod:`paper_curator.generation.citations` verifies the result rather than
trusting it.
"""

from __future__ import annotations

from paper_curator.retrieval.context import AssembledContext

PROMPT_VERSION = "v1"
"""Bumped on every change to the text below. Recorded on each answer."""

# S105 flags any name containing "TOKEN" as a possible hardcoded credential.
# This is a sentinel the model is asked to emit, and it is deliberately public:
# it appears verbatim in the system prompt below.
REFUSAL_TOKEN = "INSUFFICIENT_EVIDENCE"  # noqa: S105
"""Exact string the model emits when the context does not support an answer."""


SYSTEM_PROMPT = f"""\
You are a research assistant that answers questions about academic papers.

You will be given numbered passages from real papers, then a question.

Rules:

1. Answer ONLY from the passages provided. Do not use anything you know from
   outside them, even if you are confident it is correct.
2. Cite every claim with the label of the passage it came from, like this:
   Retrieval-augmented generation reduces hallucination [P1], though it
   depends on retrieval quality [P2].
3. Use only labels that appear in the passages. Never invent a label.
4. If several passages support a claim, cite them all: [P1][P3].
5. If the passages do not contain enough information to answer, reply with
   exactly this and nothing else:

{REFUSAL_TOKEN}

6. Do not speculate, and do not add a conclusion that goes beyond the
   passages. A short answer that is fully supported is better than a long one
   that is partly guessed.
7. Write in plain prose. Do not restate the question.
"""


USER_TEMPLATE = """\
Passages:

{context}

---

Question: {question}

Answer the question using only the passages above, citing each claim with its \
label.
"""


def build_messages(question: str, context: AssembledContext) -> list[dict[str, str]]:
    """Build the chat messages for one question.

    Raises:
        ValueError: The context is empty. Generating from no evidence is the
            failure this system exists to prevent, so it is refused here
            rather than left to the model's judgement -- a model given no
            passages and asked a question will usually answer it from memory,
            fluently and without any way to tell.

    """
    if context.is_empty:
        msg = "cannot build a prompt from empty context"
        raise ValueError(msg)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(context=context.text, question=question.strip()),
        },
    ]


def is_refusal(text: str) -> bool:
    """Report whether the model signalled that the evidence was insufficient.

    Checks for the sentinel anywhere in a short response rather than requiring
    an exact match, because models reliably emit the token and unreliably
    resist adding a courteous sentence around it. The length bound keeps a
    genuine answer that happens to mention the token from being discarded.
    """
    stripped = text.strip()
    return REFUSAL_TOKEN in stripped and len(stripped) < len(REFUSAL_TOKEN) + 200
