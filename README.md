# AI Research Paper Curator

[![CI](https://github.com/Pallavi-Nile-98/AI-Research-Paper-Curator_RAG-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Pallavi-Nile-98/AI-Research-Paper-Curator_RAG-System/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

A production-oriented Retrieval-Augmented Generation system that ingests papers
from arXiv, indexes them for both keyword and semantic search, and answers
technical questions with **grounded, source-cited responses**.

> **Status: Phase 0 of 7 complete.** Foundations, containerised services and
> architecture decisions are in place and verified. The ingestion pipeline,
> retrieval layer and API are not built yet. See
> [Project status](#project-status) for exactly what does and does not work
> today.

---

## The problem

Roughly 20,000 papers are submitted to arXiv every month. Staying current in a
single subfield means reading more than anyone has time for.

General-purpose chatbots answer questions about research confidently and
sometimes wrongly — inventing plausible papers, misattributing findings, and
citing results that do not exist. The failure is not that the model is bad; it
is that the model is answering from memory rather than from evidence.

This system answers only from passages it actually retrieved, cites the specific
paper and section behind each claim, and **states plainly when the retrieved
evidence is insufficient** rather than filling the gap.

## Intended users

- **Researchers and graduate students** tracking a fast-moving subfield.
- **Engineers** evaluating whether a technique is worth adopting, who need the
  source rather than a summary.
- **Anyone who needs a citation they can check**, not a confident paragraph.

## Key capabilities

*Planned scope. Items marked ✅ are built and verified; the rest are not yet
implemented.*

| Capability | Status |
|---|---|
| Configuration, structured logging, error hierarchy | ✅ Built, 40 tests passing |
| Containerised PostgreSQL + OpenSearch with health checks | ✅ Built and verified running |
| Architecture decision records | ✅ 7 written |
| CI: lint, format, type check, tests | ✅ Built |
| Idempotent arXiv ingestion with version tracking | ⬜ Phase 1 |
| PDF extraction with OCR fallback for scanned pages | ⬜ Phase 1 |
| Structure-aware chunking that respects paper sections | ⬜ Phase 1 |
| Hybrid retrieval — BM25 + dense vectors, fused | ⬜ Phase 2 |
| Cross-encoder re-ranking | ⬜ Phase 2 |
| **Measured retrieval benchmarks** (Precision@K, Recall@K, MRR, nDCG@K) | ⬜ Phase 2 |
| Grounded generation with validated citations | ⬜ Phase 3 |
| Async FastAPI service | ⬜ Phase 3 |
| React + TypeScript client | ⬜ Phase 3 |
| Langfuse tracing and RAGAS evaluation | ⬜ Phase 4 |
| AWS architecture and Terraform modules | ⬜ Phase 5 |

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.13 | Type hints throughout, checked with mypy in strict mode |
| Relational store | PostgreSQL 16 | Pipeline state, idempotency constraints, user feedback — [ADR-0002](docs/adr/0002-postgresql-and-opensearch-responsibilities.md) |
| Search | OpenSearch 2.19 | BM25 and k-NN vectors in one document — [ADR-0002](docs/adr/0002-postgresql-and-opensearch-responsibilities.md) |
| Retrieval | BM25 + dense, fused with RRF | Uncorrelated failure modes — [ADR-0004](docs/adr/0004-hybrid-retrieval-with-bm25-and-dense-vectors.md) |
| Embeddings | `BAAI/bge-small-en-v1.5` | 384-dim, fast on CPU |
| Re-ranking | `ms-marco-MiniLM-L-6-v2` cross-encoder | Precision pass over a wide candidate pool |
| LLM serving | Ollama, local open-weights | Free, reproducible evaluation — [ADR-0003](docs/adr/0003-ollama-for-local-llm-serving.md) |
| API | FastAPI + Pydantic | Async I/O, validated schemas, generated OpenAPI |
| Frontend | React + TypeScript | Typed API client |
| Orchestration | Apache Airflow | Scheduled, retryable ingestion |
| Observability | Langfuse (hosted) | Per-stage tracing, prompt versioning — [ADR-0007](docs/adr/0007-langfuse-cloud-rather-than-self-hosted.md) |
| Evaluation | RAGAS + custom retrieval metrics | Generation quality and retrieval quality measured separately |
| Local services | Docker Compose, profiled | Fits a 16 GB machine — [ADR-0006](docs/adr/0006-compose-profiles-for-constrained-local-development.md) |
| CI | GitHub Actions | Same pre-commit hooks as local, no drift |

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.12+** | 3.13 recommended; matches CI and the Docker images |
| **Docker Desktop** | Must be running. Allocate at least 6 GB to its VM |
| **Git** | — |
| Node.js 20+ | Only needed from Phase 3, for the React client |

Not needed on your machine: Ollama, Tesseract and Poppler all run inside
containers.

## Getting started

**1. Clone and enter the repository**

```bash
git clone https://github.com/Pallavi-Nile-98/AI-Research-Paper-Curator_RAG-System.git
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Install the package — this step is required, not optional**

```bash
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

> On macOS or Linux use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.
>
> **Do not skip this.** Source lives under `src/`, so without an install you get
> `ModuleNotFoundError: No module named 'paper_curator'`. This is a deliberate
> tradeoff explained in [ADR-0005](docs/adr/0005-src-layout-and-package-boundaries.md):
> it guarantees tests exercise the installed package rather than loose files,
> so a packaging mistake fails in CI instead of in a container.

**4. Check your environment**

```bash
.venv/Scripts/python.exe scripts/check_env.py
```

Reports what is installed and which ports are free. Exits non-zero if anything
required is missing.

**5. Create your local configuration (optional)**

```bash
copy .env.example .env
```

Every setting has a working default, so the project runs without a `.env`.
Create one only to override something.

**6. Install the git hooks (recommended)**

```bash
.venv/Scripts/python.exe -m pre_commit install
```

Runs ruff, mypy and safety checks before each commit.

## Running the local services

**Start the core stack** — PostgreSQL and OpenSearch, roughly 2.5 GB:

```bash
docker-compose up -d
```

**Watch until both report `healthy`** — Postgres takes ~30s, OpenSearch ~50s:

```bash
docker-compose ps
```

> ⚠️ `docker-compose up -d` deliberately does **not** start everything. Optional
> services sit behind profiles so a 16 GB machine is not asked for 13 GB of
> containers. See [ADR-0006](docs/adr/0006-compose-profiles-for-constrained-local-development.md).

| To also start | Command |
|---|---|
| OpenSearch Dashboards | `docker-compose --profile search-ui up -d` |
| Ollama (LLM serving) | `docker-compose --profile llm up -d` |

**Stop, keeping data:**

```bash
docker-compose down
```

**Stop and delete all data — irreversible:**

```bash
docker-compose down -v
```

## Running the tests

**Unit tests** (fast, no services required):

```bash
.venv/Scripts/python.exe -m pytest -m "not integration and not evaluation"
```

**Everything, with coverage:**

```bash
.venv/Scripts/python.exe -m pytest --cov --cov-report=term-missing
```

**Lint, format and type checks** (the same checks CI runs):

```bash
.venv/Scripts/python.exe -m pre_commit run --all-files
```

## Project status

**Phase 0 of 7 complete.**

### Verified working

Each item below was run, not assumed:

- **40 unit tests pass**; `ruff check`, `ruff format --check` and `mypy --strict`
  are all clean across 18 source files.
- **PostgreSQL 16.15 container** reaches `healthy` in ~30s, with
  `paper_curator`, `paper_curator_test` and `airflow` created by the init script.
- **OpenSearch 2.19.6 container** reaches `healthy` in ~50s, reports cluster
  status `green`, and has the `opensearch-knn` and `opensearch-neural-search`
  plugins available — confirming dense-vector retrieval is possible.
- **All 12 pre-commit hooks pass** against every file.

### Not yet verified

- The `search-ui`, `llm` and `airflow` Compose profiles have never been started.
  Their memory figures in [ADR-0006](docs/adr/0006-compose-profiles-for-constrained-local-development.md)
  are estimates, clearly labelled as such.
- No evaluation numbers exist yet. **No performance claim appears anywhere in
  this repository, and none will until it is backed by a reproducible
  experiment.**

### Roadmap

| Phase | Scope |
|---|---|
| 0 ✅ | Scaffolding, configuration, logging, containers, ADRs, CI |
| 1 ⬜ | Data model, arXiv ingestion, PDF extraction, chunking, indexing |
| 2 ⬜ | Retrieval, re-ranking, labelled dataset, measured experiments |
| 3 ⬜ | Prompts, Ollama, FastAPI, React client |
| 4 ⬜ | Langfuse tracing, RAGAS evaluation, feedback |
| 5 ⬜ | AWS architecture, Terraform, expanded CI/CD |
| 6–7 ⬜ | Documentation, portfolio materials, final verification |

## Architecture decisions

Every significant decision is recorded in [`docs/adr/`](docs/adr/), including
the alternatives rejected and the **drawbacks accepted**:

| ADR | Decision |
|---|---|
| [0001](docs/adr/0001-record-architecture-decisions.md) | Why decisions are recorded, and why records are never edited |
| [0002](docs/adr/0002-postgresql-and-opensearch-responsibilities.md) | Two datastores: PostgreSQL is truth, OpenSearch is a derived index |
| [0003](docs/adr/0003-ollama-for-local-llm-serving.md) | A local open-weights model instead of a hosted API |
| [0004](docs/adr/0004-hybrid-retrieval-with-bm25-and-dense-vectors.md) | Hybrid retrieval — stated as a hypothesis to be measured |
| [0005](docs/adr/0005-src-layout-and-package-boundaries.md) | `src/` layout and package boundaries |
| [0006](docs/adr/0006-compose-profiles-for-constrained-local-development.md) | Compose profiles for a memory-constrained machine |
| [0007](docs/adr/0007-langfuse-cloud-rather-than-self-hosted.md) | Hosted Langfuse, and the privacy tradeoff it carries |

## Security notes

- **The OpenSearch container runs with its security plugin disabled** — no TLS,
  no authentication. This is acceptable only because the port is published to
  localhost on a development machine. The AWS design uses Amazon OpenSearch
  Service with TLS and fine-grained access control.
- No secrets are committed. `.env` is git-ignored; only `.env.example` with
  placeholder values is tracked.
- `detect-private-key` and `check-added-large-files` run before every commit.
- Non-standard local ports — Postgres on `5433`, API on `8002` — because `5432`
  and `8000` are commonly occupied.

## Licence and attribution

Source code is [MIT licensed](LICENSE). This covers the code only:

- **arXiv content** is subject to arXiv's Terms of Use and each paper's own
  licence. Papers are downloaded at runtime and never committed here.
- **Model weights** are downloaded at runtime from Ollama and Hugging Face under
  their own licences, **some of which are not OSI-approved**. A licence review is
  scheduled before this README recommends a specific default model.
