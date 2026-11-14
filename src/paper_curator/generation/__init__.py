"""Turning retrieved context into a grounded, cited answer.

* ``prompts``   -- versioned prompt text; :data:`~paper_curator.generation.prompts.PROMPT_VERSION`
  is recorded on every answer, so a quality regression can be attributed to a
  specific change
* ``llm``       -- async Ollama client with timeouts, classified errors and bounded
  retries, plus a scripted fake so the pipeline is testable without a model server
* ``citations`` -- extracts the labels an answer cited and checks them against the
  sources it was actually given
* ``service``   -- orchestration: retrieve, assemble, generate, verify

The design point worth knowing: the prompt instructs the model to cite only
supplied sources, and ``citations`` verifies that it did. The instruction alone
is not a guarantee -- a small model will occasionally cite a label that was
never supplied, in a sentence that reads exactly like every other sentence.
Checking afterwards is what makes "grounded" a measured property rather than a
hope.
"""

from __future__ import annotations

from paper_curator.generation.citations import classify, extract_labels, strip_invented
from paper_curator.generation.errors import (
    LLMTimeoutError,
    LLMUnavailableError,
    MalformedResponseError,
    ModelNotFoundError,
)
from paper_curator.generation.llm import (
    LLMClient,
    OllamaClient,
    ScriptedLLM,
    create_llm_client,
)
from paper_curator.generation.models import (
    GeneratedAnswer,
    LLMResponse,
    RefusalReason,
    TokenUsage,
)
from paper_curator.generation.prompts import PROMPT_VERSION, REFUSAL_TOKEN, build_messages
from paper_curator.generation.service import AnswerService, Retriever

__all__ = [
    "PROMPT_VERSION",
    "REFUSAL_TOKEN",
    "AnswerService",
    "GeneratedAnswer",
    "LLMClient",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "MalformedResponseError",
    "ModelNotFoundError",
    "OllamaClient",
    "RefusalReason",
    "Retriever",
    "ScriptedLLM",
    "TokenUsage",
    "build_messages",
    "classify",
    "create_llm_client",
    "extract_labels",
    "strip_invented",
]
