"""Environment-driven application configuration.

Every tunable value in the system is declared here as a typed, validated field with a
safe default. Nothing reads ``os.environ`` directly anywhere else in the codebase.

Design notes
------------
* Config is grouped into small ``BaseSettings`` classes, one per external system, each
  with its own environment-variable prefix (``POSTGRES_``, ``OPENSEARCH_``, ...). This
  keeps the ``.env`` file self-documenting and lets a single subsystem be configured in
  tests without constructing the whole object graph.
* Secrets use ``SecretStr``, whose ``repr`` renders as ``**********``. A stray log line
  or traceback therefore cannot leak a password.
* ``get_settings()`` is cached, so the environment is parsed exactly once per process.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Any, Literal
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["local", "ci", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["console", "json"]

# A list field that pydantic-settings must NOT JSON-decode itself.
#
# By default the environment source calls json.loads() on any value destined for a
# complex type, and it does so *before* field validators run -- so a plain
# `ARXIV_CATEGORIES=cs.CL,cs.IR` blows up with a JSONDecodeError that no validator ever
# gets to intercept. `NoDecode` hands the raw string to `_split_csv` instead, which
# accepts both the JSON and the comma-separated form.
StringList = Annotated[list[str], NoDecode]


def _split_csv(value: Any) -> Any:
    """Accept either a JSON array or a plain comma-separated string for list fields.

    Writing ``ARXIV_CATEGORIES=["cs.CL","cs.IR"]`` in a ``.env`` file is awkward and easy
    to get wrong, so ``ARXIV_CATEGORIES=cs.CL,cs.IR`` is accepted as well. A malformed
    JSON array raises ``JSONDecodeError``, a ``ValueError`` subclass, which pydantic
    surfaces as a normal validation error.
    """
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        return json.loads(stripped)
    return [item.strip() for item in stripped.split(",") if item.strip()]


class _BaseConfig(BaseSettings):
    """Shared settings behaviour: read ``.env``, ignore unrelated variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class AppSettings(_BaseConfig):
    """Top-level application behaviour."""

    model_config = SettingsConfigDict(env_prefix="APP_")

    environment: Environment = "local"
    debug: bool = False
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "console"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


class PostgresSettings(_BaseConfig):
    """Connection settings for the relational store.

    Postgres owns pipeline state and relational truth: papers, authors, ingestion runs,
    checkpoints, processing status, queries and feedback. It is deliberately *not* the
    search engine -- see docs/adr/0002.
    """

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    host: str = "localhost"
    # 5433, not 5432: a native Postgres already occupies 5432 on the dev machine, so the
    # container publishes to 5433 on the host. Inside Docker the port is still 5432.
    port: int = Field(default=5433, ge=1, le=65535)
    user: str = "curator"
    password: SecretStr = SecretStr("curator_local_dev")
    database: str = "paper_curator"
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    echo_sql: bool = False

    def _dsn(self, driver: str) -> str:
        password = quote_plus(self.password.get_secret_value())
        user = quote_plus(self.user)
        return f"postgresql+{driver}://{user}:{password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_dsn(self) -> str:
        """SQLAlchemy URL for the async application runtime."""
        return self._dsn("asyncpg")

    @property
    def sync_dsn(self) -> str:
        """SQLAlchemy URL for synchronous tooling (Alembic migrations)."""
        return self._dsn("psycopg")


class OpenSearchSettings(_BaseConfig):
    """Connection and index settings for the search engine.

    OpenSearch owns the retrieval surface: BM25 lexical scoring and dense-vector k-NN
    over the same documents.
    """

    model_config = SettingsConfigDict(env_prefix="OPENSEARCH_")

    host: str = "localhost"
    port: int = Field(default=9200, ge=1, le=65535)
    use_ssl: bool = False
    verify_certs: bool = False
    username: str | None = None
    password: SecretStr | None = None
    # Reads and writes target an alias, never a concrete index. Re-indexing under a new
    # mapping is then an atomic alias swap instead of downtime. See docs/adr/0002.
    index_alias: str = "papers-current"
    index_prefix: str = "papers"
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    bulk_batch_size: int = Field(default=100, ge=1)

    @property
    def url(self) -> str:
        scheme = "https" if self.use_ssl else "http"
        return f"{scheme}://{self.host}:{self.port}"


class OllamaSettings(_BaseConfig):
    """Local LLM serving via Ollama."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    # Generous default: this machine has no CUDA GPU, so generation is CPU-bound.
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1)


class EmbeddingSettings(_BaseConfig):
    """Dense embedding model used for vector retrieval.

    ``dimensions`` must match the ``knn_vector`` dimension in the OpenSearch mapping.
    Changing the model therefore requires a re-index under a new index version.
    """

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    provider: Literal["sentence_transformers"] = "sentence_transformers"
    model_name: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = Field(default=384, ge=1)
    batch_size: int = Field(default=32, ge=1)
    device: str = "cpu"
    normalize: bool = True


class RerankerSettings(_BaseConfig):
    """Cross-encoder re-ranking applied to the fused candidate pool."""

    model_config = SettingsConfigDict(env_prefix="RERANKER_")

    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    batch_size: int = Field(default=16, ge=1)
    # Retrieve a wide pool, re-rank it, then keep the best few.
    candidate_pool_size: int = Field(default=50, ge=1)
    top_k: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=10.0, gt=0)


class ArxivSettings(_BaseConfig):
    """arXiv API client behaviour."""

    model_config = SettingsConfigDict(env_prefix="ARXIV_")

    base_url: str = "http://export.arxiv.org/api/query"
    categories: StringList = Field(default_factory=lambda: ["cs.CL", "cs.IR", "cs.LG"])
    page_size: int = Field(default=100, ge=1, le=2000)
    max_results_per_run: int = Field(default=200, ge=1)
    # arXiv's terms of use ask for at least three seconds between requests.
    min_request_interval_seconds: float = Field(default=3.0, ge=0)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    max_pdf_bytes: int = Field(default=50 * 1024 * 1024, ge=1)

    _split_categories = field_validator("categories", mode="before")(_split_csv)


class LangfuseSettings(_BaseConfig):
    """Optional LLM observability.

    Tracing must never be load-bearing: if credentials are absent the application runs
    normally with tracing disabled. Traces are sent to Langfuse Cloud, so the tracing
    layer must not attach secrets or unnecessary personal data.
    """

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_")

    public_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    host: str = "https://us.cloud.langfuse.com"

    @property
    def enabled(self) -> bool:
        """True only when both credentials are present."""
        return self.public_key is not None and self.secret_key is not None


class ApiSettings(_BaseConfig):
    """FastAPI service configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    # Loopback by default. The container overrides this to 0.0.0.0 explicitly, so
    # binding to every interface is always a deliberate act rather than an accident.
    host: str = "127.0.0.1"
    # 8002, not 8000: both 8000 and 8001 are already occupied on this dev machine.
    # Override with API_PORT anywhere else; scripts/check_env.py flags conflicts.
    port: int = Field(default=8002, ge=1, le=65535)
    cors_origins: StringList = Field(default_factory=lambda: ["http://localhost:5173"])
    max_request_bytes: int = Field(default=1_048_576, ge=1)
    rate_limit_per_minute: int = Field(default=60, ge=1)
    request_timeout_seconds: float = Field(default=180.0, gt=0)

    _split_origins = field_validator("cors_origins", mode="before")(_split_csv)


class Settings(_BaseConfig):
    """Root settings object aggregating every subsystem."""

    app: AppSettings = Field(default_factory=AppSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the environment is parsed once. Tests that manipulate environment
    variables must call ``get_settings.cache_clear()`` afterwards.
    """
    return Settings()
