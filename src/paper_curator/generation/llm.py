"""Talking to the language model.

A narrow protocol -- :class:`LLMClient` -- with two implementations: one that
calls Ollama over HTTP, and one that returns scripted text with no network at
all. Everything downstream depends on the protocol, so the whole generation
pipeline, including the grounding checks, is testable without a model server
and without a 2 GB download in CI.

The fake is not a convenience. The behaviour most worth testing is what happens
when the model misbehaves -- cites a source that was never supplied, runs past
the token limit, emits a refusal -- and a real model produces those cases
occasionally and unpredictably. :class:`ScriptedLLM` produces them on demand.

**Errors are classified before they are retried.** A timeout is ordinary on CPU
inference and worth another attempt; a missing model fails identically forever,
and retrying it three times only delays the message telling the user to pull it.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import httpx

from paper_curator.core.config import Settings, get_settings
from paper_curator.core.logging import get_logger
from paper_curator.core.retry import retry_async
from paper_curator.generation.errors import (
    LLMTimeoutError,
    LLMUnavailableError,
    MalformedResponseError,
    ModelNotFoundError,
)
from paper_curator.generation.models import LLMResponse, TokenUsage

logger = get_logger(__name__)


class LLMClient(Protocol):
    """Anything that can turn chat messages into a completion."""

    async def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Generate one completion."""
        ...

    async def health(self) -> bool:
        """Report whether the backend is reachable and has the configured model."""
        ...


class OllamaClient:
    """Async client for Ollama's chat endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        self._config = resolved.ollama
        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=httpx.Timeout(self._config.request_timeout_seconds),
        )

    @property
    def model(self) -> str:
        return self._config.model

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def health(self) -> bool:
        """Check the server is up and the configured model is present.

        Returns False rather than raising, because this is used to decide
        whether to start work. Distinguishing *why* it failed is the job of
        :meth:`complete`, which has a real request to attribute the failure to.
        """
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
            names = {model.get("name", "") for model in response.json().get("models", [])}
        except (httpx.HTTPError, ValueError, KeyError):
            return False

        # Ollama reports "llama3.2:3b"; a configured bare "llama3.2" should
        # still match the tagged name the server actually holds.
        return any(name == self.model or name.startswith(f"{self.model}:") for name in names)

    def _parse(self, payload: dict[str, Any], duration_ms: int) -> LLMResponse:
        """Turn Ollama's response body into an :class:`LLMResponse`."""
        message = payload.get("message")
        if not isinstance(message, dict) or "content" not in message:
            msg = "Ollama response contained no message content"
            raise MalformedResponseError(msg, context={"keys": sorted(payload)})

        return LLMResponse(
            text=str(message["content"]),
            model=str(payload.get("model", self.model)),
            usage=TokenUsage(
                prompt_tokens=int(payload.get("prompt_eval_count", 0)),
                completion_tokens=int(payload.get("eval_count", 0)),
            ),
            # Ollama says "length" when generation stopped at num_predict.
            finish_reason=str(payload.get("done_reason", "stop")),
            duration_ms=duration_ms,
        )

    async def _request(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Issue one chat request, classifying failures by what they mean."""
        started = time.perf_counter()
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                # Low, not zero. Zero is greedy decoding, which on a small
                # model tends to repeat phrases; a little randomness avoids
                # that while keeping output close to deterministic.
                "temperature": self._config.temperature,
                "num_predict": self._config.max_output_tokens,
            },
        }

        try:
            response = await self._client.post("/api/chat", json=body)
        except httpx.TimeoutException as exc:
            msg = f"Ollama did not respond within {self._config.request_timeout_seconds}s"
            raise LLMTimeoutError(msg, context={"model": self.model}) from exc
        except httpx.HTTPError as exc:
            msg = f"could not reach Ollama at {self._config.base_url}"
            raise LLMUnavailableError(msg, context={"model": self.model}) from exc

        if response.status_code == httpx.codes.NOT_FOUND:
            # The server answered; it simply does not have this model. Not a
            # transient failure, so it must not be retried.
            msg = f"model {self.model!r} is not available on the Ollama server"
            raise ModelNotFoundError(msg, context={"hint": f"run: ollama pull {self.model}"})

        if response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
            msg = f"Ollama returned {response.status_code}"
            raise LLMUnavailableError(msg, context={"body": response.text[:200]})

        if response.status_code >= httpx.codes.BAD_REQUEST:
            msg = f"Ollama rejected the request with {response.status_code}"
            raise MalformedResponseError(msg, context={"body": response.text[:200]})

        try:
            payload = response.json()
        except ValueError as exc:
            msg = "Ollama returned a body that was not JSON"
            raise MalformedResponseError(msg, context={"body": response.text[:200]}) from exc

        return self._parse(payload, int((time.perf_counter() - started) * 1000))

    async def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Generate a completion, retrying only transient failures."""
        result = await retry_async(
            lambda: self._request(messages),
            max_attempts=self._config.max_retries + 1,
            retry_on=(LLMUnavailableError, LLMTimeoutError),
            base_delay=1.0,
            operation_name="ollama_chat",
        )
        logger.debug(
            "llm_completed",
            model=result.model,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            duration_ms=result.duration_ms,
            finish_reason=result.finish_reason,
        )
        return result


class ScriptedLLM:
    """A fake model that returns prepared responses.

    Lets tests exercise the failure modes that matter -- invented citations, a
    refusal, output cut off at the token limit -- deterministically, instead of
    hoping a real model produces them.
    """

    def __init__(
        self,
        responses: list[str | LLMResponse] | None = None,
        *,
        model: str = "scripted",
        healthy: bool = True,
    ) -> None:
        self._responses = list(responses or ["A grounded answer [P1]."])
        self._model = model
        self._healthy = healthy
        self.calls: list[list[dict[str, str]]] = []
        """Every message list received, for asserting on the prompt itself."""

    async def health(self) -> bool:
        return self._healthy

    async def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Return the next scripted response, repeating the last one forever."""
        self.calls.append(messages)
        item = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(
            text=item,
            model=self._model,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
        )


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Build the configured LLM client."""
    return OllamaClient(settings)
