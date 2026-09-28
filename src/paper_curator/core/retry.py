"""Bounded retry with exponential backoff and jitter.

Used for every outbound call: arXiv, OpenSearch, Ollama. Written here rather
than pulled from a library because the policy is small, and the two properties
that make it correct are worth having visible in the codebase.

**Only retry what retrying can fix.** A timeout or a 503 may succeed on a second
attempt. A 400 will not, and retrying it wastes time and hammers a service that
already said no. Callers pass the exception types that are genuinely transient;
everything else propagates immediately.

**Jitter is not optional.** With plain exponential backoff, every client that
failed at the same moment retries at the same moment — the original failure is
followed by a synchronised burst that causes the next one. Randomising each
delay spreads the retries out. This implementation uses "full jitter": a uniform
draw from ``[0, computed_delay]`` rather than the delay itself.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from paper_curator.core.logging import get_logger

logger = get_logger(__name__)


def compute_backoff(
    attempt: int,
    *,
    base_delay: float,
    max_delay: float,
    jitter: bool = True,
) -> float:
    """Return the delay in seconds before ``attempt`` (0-based).

    Doubles per attempt up to ``max_delay``, then applies full jitter. Exposed
    separately from :func:`retry_async` so the schedule can be tested
    deterministically by disabling jitter.
    """
    # 2.0 rather than 2: int.__pow__ is typed as returning Any, which would
    # make this function return Any despite its float annotation.
    delay = min(base_delay * (2.0**attempt), max_delay)
    if not jitter:
        return delay
    # Not a security-sensitive draw -- this only decorrelates retry timing
    # between clients, so the standard PRNG is appropriate.
    return random.uniform(0, delay)  # noqa: S311


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    retry_on: tuple[type[Exception], ...],
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    operation_name: str = "operation",
) -> T:
    """Call ``operation`` until it succeeds or the attempt budget is exhausted.

    Args:
        operation: A zero-argument coroutine function. Use ``functools.partial``
            or a closure to bind arguments.
        max_attempts: Total attempts including the first. ``1`` disables retrying.
        retry_on: Exception types considered transient. Anything else propagates
            on the first occurrence.
        base_delay: Seconds before the first retry, doubling thereafter.
        max_delay: Ceiling applied before jitter.
        jitter: Randomise delays. Disable only in tests.
        operation_name: Included in log records to identify what was retried.

    Returns:
        Whatever ``operation`` returns.

    Raises:
        Exception: The last exception raised, once attempts are exhausted. The
            original is re-raised rather than wrapped, so callers can still
            inspect status codes and other attributes.

    """
    if max_attempts < 1:
        msg = f"max_attempts must be at least 1, got {max_attempts}"
        raise ValueError(msg)

    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await operation()
        except retry_on as exc:
            last_error = exc
            is_final = attempt == max_attempts - 1

            if is_final:
                logger.warning(
                    "retry_exhausted",
                    operation=operation_name,
                    attempts=max_attempts,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise

            delay = compute_backoff(
                attempt, base_delay=base_delay, max_delay=max_delay, jitter=jitter
            )
            logger.info(
                "retry_scheduled",
                operation=operation_name,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                delay_seconds=round(delay, 3),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            await asyncio.sleep(delay)

    # Unreachable: the final attempt either returns or re-raises above. Present
    # so the function is provably total to a type checker.
    raise last_error  # type: ignore[misc]  # pragma: no cover


class AsyncRateLimiter:
    """Enforces a minimum interval between operations.

    arXiv's terms of use ask for at least three seconds between API requests.
    That is a courtesy limit on a free public service run by a university, not a
    performance parameter, and this project honours it.

    The lock matters. Without it, concurrent callers all read the same
    ``_last_call`` before any of them updates it, compute the same zero wait, and
    fire simultaneously — which is exactly the behaviour the limiter exists to
    prevent.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        if min_interval_seconds < 0:
            msg = f"min_interval_seconds must be non-negative, got {min_interval_seconds}"
            raise ValueError(msg)
        self._min_interval = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last_call: float | None = None

    async def acquire(self) -> None:
        """Wait until enough time has passed since the previous acquisition."""
        if self._min_interval <= 0:
            return

        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()

            if self._last_call is not None:
                elapsed = now - self._last_call
                remaining = self._min_interval - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
                    now = loop.time()

            self._last_call = now
