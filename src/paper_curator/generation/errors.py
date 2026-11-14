"""Failures specific to generation.

The distinction that matters here is **retryable or not**. A timeout on a
CPU-bound local model is ordinary and worth retrying; a request for a model
that was never pulled will fail identically forever, and retrying it three
times only delays the error message that would have told the user to run
``ollama pull``.
"""

from __future__ import annotations

from paper_curator.core.exceptions import GenerationError


class LLMUnavailableError(GenerationError):
    """The model server could not be reached.

    Retryable. Usually means the container is not running, or is still
    loading a model into memory.
    """


class LLMTimeoutError(GenerationError):
    """The model did not respond within the configured timeout.

    Retryable, but worth surfacing rather than hiding: on CPU inference a
    timeout usually means the context was too long for the hardware, not that
    anything is broken, and the fix is a smaller context rather than a longer
    wait.
    """


class ModelNotFoundError(GenerationError):
    """The requested model is not present on the server.

    Not retryable. Deliberately separate from :class:`LLMUnavailableError`,
    because the server answered perfectly well -- it just does not have this
    model, and the fix is a pull rather than a wait.
    """


class MalformedResponseError(GenerationError):
    """The server returned something that was not a usable completion.

    Not retryable by default. A response missing the expected fields means
    the API contract changed or a proxy intervened, and repeating the request
    produces the same shape again.
    """
