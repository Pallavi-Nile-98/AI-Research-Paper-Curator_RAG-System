"""Tests for environment-driven configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_curator.core.config import Settings, get_settings


@pytest.mark.unit
def test_defaults_load_with_no_environment(isolated_settings_env: None) -> None:
    """The system must be runnable with an empty environment."""
    settings = Settings()

    assert settings.app.environment == "local"
    assert settings.app.log_format == "console"
    # Non-standard ports: 5432, 8000 and 8001 are already occupied on this host.
    assert settings.postgres.port == 5433
    assert settings.api.port == 8002
    assert settings.opensearch.index_alias == "papers-current"
    assert settings.embedding.dimensions == 384


@pytest.mark.unit
def test_environment_variables_override_defaults(
    isolated_settings_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("APP_ENVIRONMENT", "production")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")

    settings = Settings()

    assert settings.postgres.host == "db.internal"
    assert settings.postgres.port == 6543
    assert settings.app.environment == "production"
    assert settings.app.is_production is True
    assert settings.ollama.model == "qwen2.5:7b"


@pytest.mark.unit
def test_secrets_are_not_exposed_by_repr(isolated_settings_env: None) -> None:
    """A password must not appear in a traceback, log line or debug dump."""
    settings = Settings()

    assert "curator_local_dev" not in repr(settings)
    assert "curator_local_dev" not in str(settings.postgres)
    # ...but it is still retrievable deliberately.
    assert settings.postgres.password.get_secret_value() == "curator_local_dev"


@pytest.mark.unit
def test_dsn_selects_correct_driver_and_escapes_password(
    isolated_settings_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Special characters in a password must not corrupt the connection URL."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss/w:rd")

    postgres = Settings().postgres

    assert postgres.async_dsn.startswith("postgresql+asyncpg://")
    assert postgres.sync_dsn.startswith("postgresql+psycopg://")
    # '@', '/' and ':' are URL-escaped rather than terminating the userinfo section.
    assert "p%40ss%2Fw%3Ard" in postgres.async_dsn
    assert postgres.async_dsn.endswith("/paper_curator")


@pytest.mark.unit
def test_list_fields_accept_comma_separated_values(
    isolated_settings_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.env` files should not require JSON syntax for simple lists."""
    monkeypatch.setenv("ARXIV_CATEGORIES", "cs.CL, cs.CV ,cs.AI")
    monkeypatch.setenv("API_CORS_ORIGINS", "http://localhost:5173,https://example.com")

    settings = Settings()

    assert settings.arxiv.categories == ["cs.CL", "cs.CV", "cs.AI"]
    assert settings.api.cors_origins == ["http://localhost:5173", "https://example.com"]


@pytest.mark.unit
def test_list_fields_still_accept_json(
    isolated_settings_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARXIV_CATEGORIES", '["cs.CL", "stat.ML"]')

    assert Settings().arxiv.categories == ["cs.CL", "stat.ML"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("POSTGRES_PORT", "99999"),  # above the valid TCP range
        ("APP_ENVIRONMENT", "staging"),  # not one of the allowed literals
        ("APP_LOG_LEVEL", "VERBOSE"),  # not a real logging level
        ("OLLAMA_TEMPERATURE", "5.0"),  # outside 0.0-2.0
        ("EMBEDDING_DIMENSIONS", "0"),  # must be >= 1
    ],
)
def test_invalid_values_fail_fast(
    isolated_settings_env: None,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    """Bad configuration must raise at startup, not surface as a runtime bug."""
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.unit
def test_langfuse_is_disabled_without_credentials(isolated_settings_env: None) -> None:
    """Observability is optional: no keys means tracing off, not a crash."""
    assert Settings().langfuse.enabled is False


@pytest.mark.unit
def test_langfuse_requires_both_keys_to_enable(
    isolated_settings_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-placeholder")
    assert Settings().langfuse.enabled is False

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-placeholder")
    assert Settings().langfuse.enabled is True


@pytest.mark.unit
def test_opensearch_url_reflects_ssl_setting(
    isolated_settings_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert Settings().opensearch.url == "http://localhost:9200"

    monkeypatch.setenv("OPENSEARCH_USE_SSL", "true")
    monkeypatch.setenv("OPENSEARCH_HOST", "search.example.com")
    monkeypatch.setenv("OPENSEARCH_PORT", "443")

    assert Settings().opensearch.url == "https://search.example.com:443"


@pytest.mark.unit
def test_api_binds_loopback_by_default(isolated_settings_env: None) -> None:
    """Binding to every interface must be an explicit choice, never a default."""
    assert Settings().api.host == "127.0.0.1"


@pytest.mark.unit
def test_get_settings_is_cached(isolated_settings_env: None) -> None:
    assert get_settings() is get_settings()


@pytest.mark.unit
def test_unknown_environment_variables_are_ignored(
    isolated_settings_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrelated variable in the shell must not break startup."""
    monkeypatch.setenv("POSTGRES_TOTALLY_MADE_UP_OPTION", "1")

    assert Settings().postgres.host == "localhost"
