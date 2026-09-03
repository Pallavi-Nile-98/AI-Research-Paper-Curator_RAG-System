"""Phase 0 smoke test: the package is installed and importable."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_package_exposes_version() -> None:
    import paper_curator

    assert paper_curator.__version__ == "0.1.0"


@pytest.mark.unit
def test_core_public_api_is_importable() -> None:
    """The names re-exported by ``paper_curator.core`` actually resolve.

    Guards against a broken ``__init__`` that only surfaces at runtime in a container.
    """
    from paper_curator.core import (
        PaperCuratorError,
        Settings,
        configure_logging,
        get_logger,
        get_settings,
    )

    assert callable(configure_logging)
    assert callable(get_logger)
    assert callable(get_settings)
    assert issubclass(PaperCuratorError, Exception)
    assert Settings is not None


@pytest.mark.unit
def test_all_subpackages_import_cleanly() -> None:
    """Every declared package imports without side effects or circular imports."""
    import importlib

    for name in (
        "paper_curator.api",
        "paper_curator.core",
        "paper_curator.db",
        "paper_curator.evaluation",
        "paper_curator.generation",
        "paper_curator.ingestion",
        "paper_curator.observability",
        "paper_curator.retrieval",
    ):
        assert importlib.import_module(name) is not None
