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


class RetrievalSettings(_BaseConfig):
    """How candidates are found and combined.

    These are the parameters Phase 2 measures. They are configuration rather
    than constants precisely so a comparison between settings is a config
    change and a recorded experiment, not a code edit.
    """

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    mode: Literal["keyword", "vector", "hybrid"] = "hybrid"

    top_k: int = Field(default=10, ge=1, le=100, description="Results returned to the caller")
    # Each retriever fetches more than top_k so fusion and re-ranking have
    # something to work with. Fusing two lists of 10 can only ever surface
    # those 10; a wider pool is what lets a document ranked 30th by one
    # retriever and 3rd by the other reach the final list.
    candidate_pool_size: int = Field(default=50, ge=1, le=500)

    # --- BM25 field weighting ---------------------------------------------
    # The index stores text twice: stemmed for recall, unstemmed for exact
    # terminology (see search/mapping.py). These decide how much each counts.
    bm25_text_boost: float = Field(default=1.0, ge=0.0)
    bm25_exact_boost: float = Field(default=1.5, ge=0.0)
    # A query term appearing in the title is strong evidence the whole paper is
    # about it, not merely that the phrase occurs somewhere.
    bm25_title_boost: float = Field(default=2.0, ge=0.0)

    # --- Fusion -------------------------------------------------------------
    fusion_method: Literal["rrf", "weighted"] = "rrf"
    # The constant in 1/(k + rank). Larger values flatten the curve, reducing
    # how much the very top ranks dominate. 60 is the value from the original
    # RRF paper and the usual default.
    rrf_k: int = Field(default=60, ge=1)
    # Used only by weighted fusion, which normalises scores before combining.
    # Ignored under RRF, which uses rank position alone.
    keyword_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- Result shaping -----------------------------------------------------
    # Caps how much of the final context one paper may occupy. Without it a
    # single highly relevant paper can fill every slot, and a question needing
    # two sources gets one source eight times.
    max_chunks_per_paper: int = Field(default=3, ge=1)
    # Near-duplicate passages waste context budget. Chunk overlap guarantees
    # some adjacent pairs share text by construction.
    deduplicate: bool = True
    duplicate_similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


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


class ExtractionSettings(_BaseConfig):
    """PDF download, text extraction and OCR fallback."""

    model_config = SettingsConfigDict(env_prefix="EXTRACTION_")

    # --- Download safety ---------------------------------------------------
    max_pdf_bytes: int = Field(default=50 * 1024 * 1024, ge=1, description="Refuse anything larger")
    download_timeout_seconds: float = Field(default=60.0, gt=0)
    download_max_retries: int = Field(default=2, ge=0)
    # PDF URLs come from the arXiv feed, which is third-party input. Restricting
    # the hosts we will fetch from means a compromised or spoofed feed cannot
    # redirect the downloader at an internal address -- a server-side request
    # forgery. Empty disables the check.
    allowed_pdf_hosts: StringList = Field(
        default_factory=lambda: ["arxiv.org", "www.arxiv.org", "export.arxiv.org"]
    )

    # --- Quality and OCR fallback -----------------------------------------
    # Below this score, extraction is considered poor. Whether OCR actually runs
    # additionally depends on the text being sparse rather than merely garbled --
    # OCR repairs a missing text layer, not a broken font encoding.
    quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    ocr_enabled: bool = True
    ocr_language: str = Field(default="eng", description="Tesseract language code")
    # 200 DPI is the usual floor for reliable OCR of body text. Higher improves
    # accuracy slightly and costs render time and memory roughly quadratically.
    ocr_dpi: int = Field(default=200, ge=72, le=600)
    # Bounds the cost of one bad document. OCR runs at seconds per page on a CPU,
    # so an unbounded 300-page scan could occupy a worker for half an hour.
    ocr_max_pages: int = Field(default=20, ge=1)
    ocr_timeout_seconds: float = Field(default=300.0, gt=0)

    _split_hosts = field_validator("allowed_pdf_hosts", mode="before")(_split_csv)


class ChunkingSettings(_BaseConfig):
    """How a paper's text is split into retrievable passages.

    These values are the main knob for retrieval quality and are deliberately
    configurable: Phase 2 tunes them by measurement rather than by intuition.
    Because the full extracted text is kept in PostgreSQL (ADR-0002),
    re-chunking with different values reads a column instead of re-downloading
    every PDF.
    """

    model_config = SettingsConfigDict(env_prefix="CHUNKING_")

    # 512 matches the input limit of the default embedding model. A chunk longer
    # than the model can read gets silently truncated, so the tail would be
    # indexed but never actually contribute to its own embedding.
    max_tokens: int = Field(default=512, ge=32, le=8192)
    # Chunks below this are usually headings, page numbers or stray fragments.
    # They match noisily and dilute retrieval.
    min_tokens: int = Field(default=50, ge=1)
    # Carried from the end of one chunk into the start of the next, so a passage
    # split across a boundary is still findable from either side.
    overlap_tokens: int = Field(default=64, ge=0)

    # References are citation lists: dense with author names and titles that
    # match many queries for the wrong reason. Excluded by default, but
    # configurable because "which papers cite X" is a legitimate question.
    include_references: bool = False
    # Appendices carry real content -- proofs, extra results, hyperparameters --
    # so they are kept by default and flagged rather than dropped.
    include_appendix: bool = True

    # Cap on chunks from one paper, so a pathological document cannot dominate
    # the index or one ingestion run.
    max_chunks_per_paper: int = Field(default=500, ge=1)

    @property
    def effective_overlap(self) -> int:
        """Overlap clamped to something smaller than a whole chunk.

        Overlap at or above max_tokens would make each chunk start where the
        previous one started, so the chunker would never advance.
        """
        return min(self.overlap_tokens, self.max_tokens // 2)


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
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the environment is parsed once. Tests that manipulate environment
    variables must call ``get_settings.cache_clear()`` afterwards.
    """
    return Settings()
