"""Tests for structured logging."""

from __future__ import annotations

import json

import pytest

from paper_curator.core.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
    new_request_id,
)


def _last_json_line(stderr: str) -> dict[str, object]:
    """Parse the final JSON log record from captured stderr."""
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    assert lines, "expected at least one log line on stderr"
    parsed = json.loads(lines[-1])
    assert isinstance(parsed, dict)
    return parsed


@pytest.mark.unit
def test_json_format_emits_machine_parseable_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole point of structured logging: fields stay fields."""
    configure_logging(level="INFO", log_format="json")

    get_logger("test").info("chunks_indexed", chunk_count=42, arxiv_id="2401.12345")

    record = _last_json_line(capsys.readouterr().err)
    assert record["event"] == "chunks_indexed"
    assert record["chunk_count"] == 42  # an int, not the string "42"
    assert record["arxiv_id"] == "2401.12345"
    assert record["level"] == "info"
    assert "timestamp" in record


@pytest.mark.unit
def test_console_format_is_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", log_format="console")

    get_logger("test").info("retrieval_started", mode="hybrid")

    stderr = capsys.readouterr().err
    assert "retrieval_started" in stderr
    assert "hybrid" in stderr


@pytest.mark.unit
def test_bound_context_appears_on_every_subsequent_record(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A request ID bound once must appear on later records without being passed along."""
    configure_logging(level="INFO", log_format="json")
    request_id = new_request_id()
    bind_request_context(request_id=request_id, route="/api/v1/ask")

    log = get_logger("test")
    log.info("retrieval_started")
    first = _last_json_line(capsys.readouterr().err)

    log.info("generation_completed")
    second = _last_json_line(capsys.readouterr().err)

    assert first["request_id"] == request_id
    assert first["route"] == "/api/v1/ask"
    assert second["request_id"] == request_id


@pytest.mark.unit
def test_clearing_context_stops_leakage_between_requests(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without this, request B would be logged under request A's correlation ID."""
    configure_logging(level="INFO", log_format="json")
    bind_request_context(request_id="request-a")

    log = get_logger("test")
    log.info("first_request_event")
    capsys.readouterr()

    clear_request_context()
    log.info("second_request_event")

    record = _last_json_line(capsys.readouterr().err)
    assert "request_id" not in record


@pytest.mark.unit
def test_level_filtering_suppresses_lower_severity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(level="WARNING", log_format="json")

    log = get_logger("test")
    log.info("below_threshold")
    log.warning("above_threshold")

    stderr = capsys.readouterr().err
    assert "below_threshold" not in stderr
    assert "above_threshold" in stderr


@pytest.mark.unit
def test_stdlib_loggers_are_routed_through_structlog(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Third-party libraries (uvicorn, SQLAlchemy, opensearch-py) use stdlib logging.

    Their output must land in the same JSON stream, otherwise half the logs in a
    container are unparseable and none of them carry the request ID.
    """
    import logging

    configure_logging(level="INFO", log_format="json")
    bind_request_context(request_id="request-xyz")

    logging.getLogger("some.third.party").info("third party message")

    record = _last_json_line(capsys.readouterr().err)
    assert record["event"] == "third party message"
    assert record["logger"] == "some.third.party"
    assert record["request_id"] == "request-xyz"


@pytest.mark.unit
def test_exception_info_is_captured(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", log_format="json")

    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("test").exception("pipeline_failed", stage="extraction")

    record = _last_json_line(capsys.readouterr().err)
    assert record["event"] == "pipeline_failed"
    assert record["stage"] == "extraction"
    assert "ValueError: boom" in str(record["exception"])


@pytest.mark.unit
def test_configure_logging_is_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    """Repeated configuration must not duplicate handlers, and so not duplicate lines."""
    configure_logging(level="INFO", log_format="json")
    configure_logging(level="INFO", log_format="json")
    configure_logging(level="INFO", log_format="json")

    get_logger("test").info("only_once")

    emitted = [line for line in capsys.readouterr().err.strip().splitlines() if line.strip()]
    assert len(emitted) == 1


@pytest.mark.unit
def test_new_request_id_is_unique() -> None:
    assert new_request_id() != new_request_id()
