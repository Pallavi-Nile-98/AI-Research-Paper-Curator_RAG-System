# ADR-0005: Use a `src/` layout with a single `paper_curator` package

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-09-24 |
| **Deciders** | Pallavi Nile |
| **Supersedes** | — |
| **Superseded by** | — |

## Context

The project was originally sketched with each concern as a top-level package at
the repository root:

```text
ai-research-paper-curator/
├── ingestion/
├── retrieval/
├── generation/
├── api/
├── observability/
├── evaluation/
└── tests/
```

This is a common layout and it reads well. Three problems emerged on closer
inspection, and a fourth appeared once the first files were written.

### 1. Generic top-level names collide

`api`, `evaluation`, and `generation` are ordinary English words. Installed into
a shared environment, a top-level package called `api` competes for that name
with anything else claiming it. The failure is not a clean error — it is the
wrong module being imported, silently.

### 2. A flat layout lets tests import the wrong code

This is the problem that actually bites. Python puts the current working
directory on `sys.path`. With packages at the repository root, running
`pytest` from that root makes `import retrieval` resolve to `./retrieval/` — the
working copy — regardless of what is installed.

That sounds harmless and is not. The consequence is a specific, recurring bug:

- Tests pass locally, because they import files directly from the checkout.
- The Docker image installs the package properly.
- Anything the packaging configuration failed to include — a missing
  `package-data` entry, a subpackage absent from `packages.find` — is simply
  absent at runtime.
- The container fails on a module the test suite never actually exercised in its
  installed form.

A `src/` layout removes the possibility. Source is not importable from the
repository root, so tests can only import the installed package. If packaging is
broken, the test suite fails immediately rather than in production.

### 3. Several root packages means several distributions, or a fragile one

Six root packages must either be six installable distributions — six versions to
keep in step, for code that ships as one unit — or one distribution reaching
across six unrelated top-level names, which packaging tools handle awkwardly.

### 4. Cross-cutting code had nowhere to live

Settings, logging setup, the exception hierarchy, and the SQLAlchemy models are
needed by *every* other concern. In the original tree there was no package that
owned them, so they would have landed in whichever concern touched them first —
and `ingestion` importing from `api` to reach the settings object inverts the
dependency direction the structure was meant to enforce.

## Decision

Place all importable code under `src/`, inside a single distribution package
named `paper_curator`:

```text
src/paper_curator/
├── core/           # settings, logging, exceptions  — depends on nothing internal
├── db/             # SQLAlchemy models, session management, repositories
├── ingestion/      # arXiv client, PDF extraction, chunking, pipelines
├── retrieval/      # keyword, vector, fusion, reranking, context assembly
├── generation/     # prompt templates, Ollama client
├── api/            # FastAPI routes, schemas, services, middleware
├── observability/  # Langfuse tracing
└── evaluation/     # retrieval metrics and experiment runners
```

Two supporting rules:

**`core/` and `db/` may not import from the other subpackages.** Dependencies
point inward. Anything that would require `core` to import from `ingestion`
indicates the code belongs elsewhere.

**Non-importable material stays at the repository root**, where it is visible
without digging through `src/`: `airflow/dags/`, `migrations/`, `frontend/`,
`infrastructure/`, `scripts/`, `tests/`, `docs/`.

### Two consequences of that split worth naming

**Evaluation is divided by kind, not by topic.** The runners and metric
implementations are code, so they live in `src/paper_curator/evaluation/`. The
datasets, experiment configurations and result reports are artifacts, so they
live in `evaluation/` at the repository root. They are the measured evidence
this project exists to produce, and a reader should find them without knowing
the package layout.

**Prompt templates ship as package data.** Versioned prompts live in
`src/paper_curator/generation/prompts/` and are declared in
`[tool.setuptools.package-data]`. A prompt outside the package would be absent
from the wheel and therefore absent from the container — the generation code
would import fine and then fail at first use.

## Consequences

### Positive

- One import root. `from paper_curator.retrieval import HybridRetriever` resolves
  identically in tests, in the API container, and inside an Airflow DAG file.
- Tests exercise the installed package, so packaging errors surface in CI rather
  than in a container.
- Airflow DAGs can import application services directly, which is what keeps
  business logic out of DAG files as required.
- `core/` gives settings, logging and exceptions an unambiguous owner, and the
  inward-only dependency rule is mechanically checkable later if it is worth
  enforcing.
- One `pyproject.toml`, one version number, one editable install.

### Negative

- **`pip install -e .` becomes mandatory before tests will run.** This is the real
  cost. A fresh clone that skips it gets `ModuleNotFoundError: No module named
  'paper_curator'`, which is an unhelpful error for a newcomer. It must be the
  first instruction in the README setup section.
- Paths are longer: `src/paper_curator/retrieval/fusion.py` rather than
  `retrieval/fusion.py`.
- The layout departs from the structure originally specified for this project.
  Anyone comparing the two sees a mismatch, which is precisely why this record
  exists.
- A single distribution means the whole package is installed even where only part
  is needed — the ingestion container carries the API code it never imports.
  Acceptable at this size; it would justify revisiting if the components were
  ever deployed and scaled independently.

### Neutral

- `py.typed` marks the package as typed under PEP 561, so type information is
  available to anything importing it.
- Editable installs are resolved by path, so edits under `src/` take effect
  without reinstalling.

## Alternatives considered

**Flat top-level packages, as originally sketched.** Rejected for the four
reasons in Context. The silent-wrong-import hazard was decisive: a failure mode
that makes tests pass while the deployment breaks is worse than one that fails
loudly.

**A single flat module or one package with no internal structure.** Rejected: the
project spans ingestion, retrieval, generation, serving and evaluation. Without
enforced boundaries these concerns grow into each other, and "no business logic in
DAG files or route handlers" becomes unenforceable.

**Separate distributions per concern** (`paper-curator-ingestion`,
`paper-curator-retrieval`, …). Rejected: correct for independently versioned,
independently released components. Here it would mean coordinating six version
numbers for code that always ships together — cost with no corresponding benefit.

**A `src/` layout with the original package names promoted to top level inside
it** (`src/ingestion/`, `src/api/`, …). Rejected: fixes the working-directory
import hazard but keeps the generic-name collision problem and still leaves
cross-cutting code without a home.

## References

- [Python Packaging User Guide — src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- PEP 561 — Distributing and Packaging Type Information
