"""Shared pytest fixtures.

The two fixtures here exist to stop tests leaking into one another. Both settings and
logging are process-global, so without explicit resets a test that sets ``POSTGRES_HOST``
would change the result of a test that runs after it -- and the failure would depend on
test ordering, which is the worst kind of flake to debug.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from paper_curator.core.config import get_settings

# Every environment-variable prefix understood by paper_curator.core.config.
_SETTINGS_ENV_PREFIXES = (
    "APP_",
    "POSTGRES_",
    "OPENSEARCH_",
    "OLLAMA_",
    "EMBEDDING_",
    "RERANKER_",
    "ARXIV_",
    "LANGFUSE_",
    "API_",
)


@pytest.fixture
def isolated_settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give a test a pristine configuration environment.

    Three sources of contamination are removed:

    1. Real environment variables (a developer's exported ``POSTGRES_HOST``).
    2. The repository's ``.env`` file -- handled by changing into an empty temp
       directory, since ``BaseSettings`` resolves ``.env`` relative to the working
       directory.
    3. The ``get_settings()`` LRU cache, which would otherwise return a value computed
       under a different environment.
    """
    for key in list(os.environ):
        if key.startswith(_SETTINGS_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)

    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Restore structlog to its default configuration around every test."""
    structlog.contextvars.clear_contextvars()
    yield
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
