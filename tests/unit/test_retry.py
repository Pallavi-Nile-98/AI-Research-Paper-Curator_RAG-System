"""Tests for the retry policy and rate limiter.

Both are used by every outbound call in the system, so a bug here degrades
arXiv ingestion, OpenSearch indexing and model generation simultaneously — and
degrades them in the hardest way to notice, as intermittent slowness rather
than a clear failure.

All tests disable jitter or use a zero interval so they stay deterministic and
fast. Nothing here sleeps for a meaningful length of time.
"""

from __future__ import annotations

import asyncio

import pytest

from paper_curator.core.retry import AsyncRateLimiter, compute_backoff, retry_async


class TransientError(Exception):
    """Transient failure used to trigger retries."""


class PermanentError(Exception):
    """Failure the policy is not configured to retry."""


@pytest.mark.unit
class TestComputeBackoff:
    def test_delay_doubles_per_attempt(self) -> None:
        delays = [
            compute_backoff(n, base_delay=1.0, max_delay=100.0, jitter=False) for n in range(5)
        ]
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_delay_is_capped(self) -> None:
        """Unbounded doubling reaches hours, long after a request is pointless."""
        delay = compute_backoff(20, base_delay=1.0, max_delay=30.0, jitter=False)
        assert delay == 30.0

    def test_jitter_stays_within_the_computed_delay(self) -> None:
        """Full jitter draws uniformly from [0, delay], never above it."""
        for _ in range(200):
            delay = compute_backoff(3, base_delay=1.0, max_delay=100.0, jitter=True)
            assert 0.0 <= delay <= 8.0

    def test_jitter_actually_varies(self) -> None:
        """Identical delays across clients cause a synchronised retry burst."""
        draws = {compute_backoff(3, base_delay=1.0, max_delay=100.0) for _ in range(50)}
        assert len(draws) > 1


@pytest.mark.unit
class TestRetryAsync:
    async def test_returns_immediately_on_success(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        result = await retry_async(
            operation, max_attempts=3, retry_on=(TransientError,), jitter=False
        )
        assert result == "ok"
        assert calls == 1

    async def test_retries_until_success(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TransientError("transient")
            return "ok"

        result = await retry_async(
            operation, max_attempts=5, retry_on=(TransientError,), base_delay=0.001, jitter=False
        )
        assert result == "ok"
        assert calls == 3

    async def test_reraises_the_original_error_when_exhausted(self) -> None:
        """The original is re-raised, not wrapped, so callers keep its attributes."""
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise TransientError("still failing")

        with pytest.raises(TransientError, match="still failing"):
            await retry_async(
                operation,
                max_attempts=3,
                retry_on=(TransientError,),
                base_delay=0.001,
                jitter=False,
            )
        assert calls == 3

    async def test_does_not_retry_an_unlisted_exception(self) -> None:
        """Retrying a 400 just repeats a request the service already rejected."""
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise PermanentError("permanent")

        with pytest.raises(PermanentError):
            await retry_async(operation, max_attempts=5, retry_on=(TransientError,), jitter=False)
        assert calls == 1

    async def test_max_attempts_of_one_disables_retrying(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise TransientError("once")

        with pytest.raises(TransientError):
            await retry_async(operation, max_attempts=1, retry_on=(TransientError,), jitter=False)
        assert calls == 1

    async def test_rejects_a_nonsensical_attempt_budget(self) -> None:
        async def operation() -> str:
            return "unused"

        with pytest.raises(ValueError, match="at least 1"):
            await retry_async(operation, max_attempts=0, retry_on=(TransientError,))


@pytest.mark.unit
class TestAsyncRateLimiter:
    async def test_first_acquisition_does_not_wait(self) -> None:
        limiter = AsyncRateLimiter(0.05)
        loop = asyncio.get_running_loop()
        start = loop.time()
        await limiter.acquire()
        assert loop.time() - start < 0.02

    async def test_second_acquisition_waits_for_the_interval(self) -> None:
        limiter = AsyncRateLimiter(0.05)
        loop = asyncio.get_running_loop()
        await limiter.acquire()
        start = loop.time()
        await limiter.acquire()
        assert loop.time() - start >= 0.04

    async def test_zero_interval_never_waits(self) -> None:
        """Lets tests exercise client code without real delays."""
        limiter = AsyncRateLimiter(0)
        loop = asyncio.get_running_loop()
        start = loop.time()
        for _ in range(20):
            await limiter.acquire()
        assert loop.time() - start < 0.05

    async def test_concurrent_callers_are_serialised(self) -> None:
        """Without the lock, concurrent callers all see the same last-call time.

        They would each compute a zero wait and fire together -- exactly what
        the limiter exists to prevent.
        """
        limiter = AsyncRateLimiter(0.03)
        loop = asyncio.get_running_loop()
        start = loop.time()
        await asyncio.gather(*(limiter.acquire() for _ in range(4)))
        # Four acquisitions means three enforced gaps.
        assert loop.time() - start >= 0.08

    def test_rejects_a_negative_interval(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            AsyncRateLimiter(-1.0)
