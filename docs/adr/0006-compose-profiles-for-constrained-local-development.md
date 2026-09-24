# ADR-0006: Group local services into Docker Compose profiles

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-09-24 |
| **Deciders** | Pallavi Nile |
| **Supersedes** | — |
| **Superseded by** | — |

## Context

This system depends on several long-running services: a relational database, a
search engine, a search UI, an LLM server, an orchestrator, and an observability
backend. The conventional local setup declares them all in one
`docker-compose.yml` and starts them together with `docker-compose up`.

That does not fit the machine this project is developed on.

### The measured constraint

The development machine has **15.7 GB of RAM total**, of which Docker Desktop's
WSL 2 virtual machine is allocated **7.6 GiB**. Windows and a browser consume a
substantial share of the remainder.

Estimated steady-state footprint if every service ran at once:

| Service | Approx. RAM |
|---|---|
| PostgreSQL | 0.25 GB |
| OpenSearch (1 GB JVM heap + Lucene off-heap) | 2.0 GB |
| OpenSearch Dashboards | 0.7 GB |
| Ollama with a 3B model loaded | 3.0 GB |
| Airflow (scheduler + API server + metadata DB) | 2.0 GB |
| Self-hosted Langfuse (web, worker, ClickHouse, Redis, MinIO, Postgres) | 3.5 GB |
| Docker Desktop / WSL 2 overhead | 1.5 GB |
| **Total** | **~13 GB** |

This exceeds what the machine can give without making Windows unusable. Two of
those rows were eliminated by decisions recorded elsewhere — Langfuse runs as a
hosted service (ADR-0007) — but the remainder still does not fit comfortably.

### The constraint is not only hardware

Most development work needs a small subset of these services:

- Writing and testing chunking logic needs **nothing** — it operates on fixtures.
- Building the arXiv client needs **nothing** running locally.
- Ingestion and indexing need **Postgres and OpenSearch**.
- Retrieval experiments need **Postgres and OpenSearch**.
- Only answer generation needs **Ollama**.
- Only pipeline orchestration needs **Airflow**.

Starting an LLM server to run a chunking unit test wastes 3 GB and around a
minute of startup for no benefit.

## Decision

Declare every service in a single `docker-compose.yml`, but assign the optional
ones to **Compose profiles** so that each is started deliberately.

| Profile | Services | Declared limit | Command |
|---|---|---|---|
| *(none)* | `postgres`, `opensearch` | 2.5 GB | `docker-compose up -d` |
| `search-ui` | `opensearch-dashboards` | +1.0 GB | `docker-compose --profile search-ui up -d` |
| `llm` | `ollama` | +6.0 GB | `docker-compose --profile llm up -d` |
| `airflow` | Airflow services (Phase 1) | +2.0 GB | `docker-compose --profile airflow up -d` |

A service with no `profiles:` key always starts; a service with one starts only
when that profile is named. Postgres and OpenSearch are therefore the default,
because almost everything needs them and nothing else is needed as often.

Two supporting decisions are recorded here because they exist for the same
reason:

**Per-service memory limits** are declared via `deploy.resources.limits.memory`,
so a runaway container cannot starve the host or its siblings.

**Healthchecks probe readiness, not liveness.** A container process starts
several seconds before it can serve traffic — Postgres accepts connections only
after initialisation, and OpenSearch needs roughly 50 seconds for the JVM and
plugins. Without a healthcheck, `depends_on` proceeds as soon as the process
exists, and dependent services fail against a server that is not yet listening.

## Consequences

### Positive

- Everyday development runs two containers instead of six, well inside the VM's
  7.6 GiB.
- Startup is faster, and the feedback loop shortens accordingly.
- Choosing a profile requires knowing which services a task actually needs —
  understanding worth having when explaining the architecture.
- It mirrors the production topology, where these are independently deployed and
  independently scaled services rather than one monolithic environment.
- Memory limits make resource expectations explicit and reviewable in the file
  itself, instead of being discovered when the host starts swapping.

### Negative

- **`docker-compose up -d` no longer starts everything.** This is the real cost.
  It violates the expectation most people bring to a Compose file, and the
  failure it produces is indirect: forgetting `--profile llm` surfaces later as a
  connection error against `localhost:11434`, not as a clear message about a
  service that was never started. Profiles and their commands must be documented
  prominently in the README.
- Profile names have to be remembered or looked up. Four is manageable; many more
  would not be.
- **Declared limits exceed the VM's memory.** Running the default and `llm`
  profiles together declares 8.5 GB against a 7.6 GiB VM. Limits are ceilings
  rather than reservations, so this works while actual usage stays below the
  total — but it removes the safety margin the limits were meant to provide.
  Ollama's 6 GB allowance is larger than a 3B model needs and should be reduced
  to about 4 GB when the LLM profile is first exercised in Phase 3.
- Compose profiles apply to `up`, `down`, `ps` and `logs` alike. Running
  `docker-compose ps` without a profile flag will not list profiled containers
  even while they are running, which is briefly confusing.

### Neutral

- Everything remains in one `docker-compose.yml`, so the full topology is legible
  in a single file regardless of what is currently running.
- Named volumes are declared outside the profiles, so data persists across
  profile changes and survives `docker-compose down`.
- Langfuse appears in no profile because it is not self-hosted at all
  (see ADR-0007).

## Verification

The default profile was started and verified on 2026-09-23:

- `postgres` reached `healthy` in roughly 30 seconds; the init script created
  `paper_curator`, `paper_curator_test` and `airflow`.
- `opensearch` reached `healthy` in roughly 50 seconds and reported cluster
  status `green`, with the `opensearch-knn` and `opensearch-neural-search`
  plugins present — confirming that dense-vector retrieval is available.

The `search-ui`, `llm` and `airflow` profiles have **not** been started yet.
Their memory figures above are estimates, not measurements, and should be
replaced with observed values once each profile is first exercised.

## Alternatives considered

**Start every service every time.** Rejected: approximately 13 GB against a
15.7 GB machine, most of it idle for most tasks.

**Multiple Compose files combined with `-f`** (`docker-compose.yml` plus
`docker-compose.llm.yml`). Rejected: profiles are the mechanism Compose provides
for exactly this, and a `-f` chain must be repeated correctly on every subsequent
command — omitting it on `down` leaves containers running.

**Run the services natively on the host instead of in containers.** Rejected:
this loses the version pinning and reproducibility that make the setup
repeatable on another machine, and removes the parity with the AWS deployment
that makes the containerised topology worth demonstrating.

**Shrink every service to fit simultaneously** (smaller JVM heap, a smaller
model). Partially adopted — OpenSearch runs a 1 GB heap rather than the default —
but rejected as a complete answer: reducing heap far enough to fit everything
would produce latency measurements that say more about the constrained heap than
about the retrieval design.

**Develop on a cloud instance sized for the full stack.** Rejected on cost. A
portfolio project should not carry a monthly bill, and this directly contradicts
the cost-control goal of the deployment phase.

## References

- [Docker Compose — Using profiles](https://docs.docker.com/compose/how-tos/profiles/)
- ADR-0002 — PostgreSQL and OpenSearch responsibilities
- ADR-0007 — Langfuse Cloud rather than self-hosted
