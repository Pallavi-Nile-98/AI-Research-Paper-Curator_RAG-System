"""Tests for the application exception hierarchy."""

from __future__ import annotations

import pytest

from paper_curator.core.exceptions import (
    ConfigurationError,
    EvaluationError,
    ExternalServiceError,
    GenerationError,
    IngestionError,
    PaperCuratorError,
    RetrievalError,
)

_ALL_ERRORS = (
    ConfigurationError,
    EvaluationError,
    ExternalServiceError,
    GenerationError,
    IngestionError,
    RetrievalError,
)


@pytest.mark.unit
@pytest.mark.parametrize("error_class", _ALL_ERRORS)
def test_every_error_shares_the_common_base(error_class: type[PaperCuratorError]) -> None:
    """API middleware catches one base class; anything else is a genuine 500."""
    assert issubclass(error_class, PaperCuratorError)
    assert issubclass(error_class, Exception)


@pytest.mark.unit
def test_message_is_preserved_without_context() -> None:
    error = IngestionError("pdf download failed")

    assert error.message == "pdf download failed"
    assert str(error) == "pdf download failed"
    assert error.context == {}


@pytest.mark.unit
def test_context_is_structured_and_rendered() -> None:
    """Context stays a queryable dict *and* renders into the human-readable string."""
    error = IngestionError(
        "pdf download failed",
        context={"arxiv_id": "2401.12345", "status_code": 503},
    )

    assert error.context == {"arxiv_id": "2401.12345", "status_code": 503}
    rendered = str(error)
    assert "pdf download failed" in rendered
    assert "arxiv_id='2401.12345'" in rendered
    assert "status_code=503" in rendered


@pytest.mark.unit
def test_external_service_error_records_which_service_failed() -> None:
    """Distinguishing a flaky search cluster from a dead model server drives retries."""
    error = ExternalServiceError(
        "request timed out",
        service="opensearch",
        context={"timeout_seconds": 30},
    )

    assert error.service == "opensearch"
    assert error.context["service"] == "opensearch"
    assert error.context["timeout_seconds"] == 30


@pytest.mark.unit
def test_errors_can_be_caught_by_the_base_class() -> None:
    with pytest.raises(PaperCuratorError) as caught:
        raise RetrievalError("fusion failed", context={"mode": "hybrid"})

    assert caught.value.context["mode"] == "hybrid"


@pytest.mark.unit
def test_context_is_copied_not_aliased() -> None:
    """Mutating the caller's dict afterwards must not rewrite the recorded failure."""
    original = {"attempt": 1}
    error = ConfigurationError("bad config", context=original)

    original["attempt"] = 99

    assert error.context["attempt"] == 1
