# ADR-0007: Use hosted Langfuse rather than self-hosting it

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-09-24 |
| **Deciders** | Pallavi Nile |
| **Supersedes** | — |
| **Superseded by** | — |

## Context

A RAG request is not one operation. It is query preprocessing, keyword
retrieval, vector retrieval, fusion, re-ranking, context assembly, prompt
rendering, model generation, and citation validation — each capable of being the
reason an answer came out wrong.

When an answer is bad, the question is *which stage failed*. Did retrieval miss
the relevant chunk? Did fusion rank it below the cut? Did the context builder
drop it to stay inside the token budget? Did the model ignore it? Ordinary
application logs answer this poorly: the stages appear as separate lines with no
enclosing structure, no shared view of one request's inputs and outputs, and no
way to compare the same question across two prompt versions.

Langfuse provides the LLM-specific pieces that generic tooling does not: nested
spans per request, prompt versioning with side-by-side comparison, token and
latency accounting per stage, and the ability to attach evaluation scores and
user feedback to the exact trace that produced an answer.

It can be self-hosted or used as a managed service. That is the decision here —
the choice of Langfuse itself is not contested.

### What self-hosting actually costs

Current Langfuse self-hosting is not a single container. A working deployment
needs the web application, a background worker, ClickHouse for trace analytics,
Redis for queueing, an S3-compatible object store, and its own PostgreSQL
instance — roughly **3.5 GB of RAM**.

Against the constraint recorded in ADR-0006 — 15.7 GB total, of which Docker's
VM gets 7.6 GiB — that is close to half the container budget, spent on a system
that **observes** the application rather than being part of it. Running it
alongside Postgres, OpenSearch and Ollama is not possible on this machine.

## Decision

Use **Langfuse's hosted service**. No Langfuse containers appear in
`docker-compose.yml`.

### Tracing is never load-bearing

This is the constraint that governs the integration, and it is not negotiable.

- With no credentials configured, the application runs normally with tracing
  disabled. `LangfuseSettings.enabled` returns true only when both the public and
  secret key are present.
- If the service is unreachable, slow, or rejecting requests, generation still
  succeeds. Trace export failures are logged and swallowed; they never propagate
  into the request path.
- No test requires Langfuse.

An observability layer that can take down the system it observes is worse than
no observability layer.

### What is sent, and what is not

Because traces leave this machine, the boundary must be explicit rather than
incidental.

**Sent:** the user's question, retrieved chunk text and metadata, the rendered
prompt, prompt version identifier, model name and parameters, model output,
per-stage latency and token counts, and error types.

**Never sent:** credentials of any kind, database connection strings, API keys,
raw environment variables, or the contents of `.env`. Free-text feedback comments
are treated as potentially sensitive and are stored in PostgreSQL rather than
attached to traces.

Retrieved chunk text is arXiv paper content, which is already public — so the
material genuinely at issue is the user's own queries.

## Consequences

### Positive

- Roughly 3.5 GB of local RAM freed — close to half the Docker VM budget —
  which is what makes running Postgres, OpenSearch and Ollama together feasible
  at all (ADR-0006).
- No operational burden from ClickHouse, Redis or an object store, none of which
  this project would otherwise need or learn anything from.
- Prompt versioning, trace comparison and dashboards work immediately, without
  being built or maintained.
- The free tier is sufficient for portfolio-scale traffic.

### Negative

- **Traces containing user queries leave the machine and are held by a third
  party.** This is the real cost of the decision. Paper content is already
  public; the queries are not. This is acceptable for a portfolio project with a
  single user and must be reconsidered before any deployment with real users,
  where self-hosting or an explicit data-processing agreement would be required.
- **It contradicts a principle stated in ADR-0003** — that the repository should
  run without anyone creating an account. The contradiction is real and is
  resolved only by tracing being optional: the project runs fully without
  Langfuse credentials, and observability is the one capability a cloner does not
  get. Worth naming rather than glossing over.
- Free-tier retention and volume limits apply. Traces older than the retention
  window disappear, so any comparison intended to survive must be exported into
  `evaluation/reports/` rather than left in the dashboard.
- Less control: the underlying trace store cannot be queried directly, only
  through the provided interface and export.
- An external dependency whose availability and terms are outside this project's
  control.

### Neutral

- The SDK and instrumentation code are identical for hosted and self-hosted
  deployments. Moving to self-hosted later changes a base URL and adds compose
  services — no application code changes.
- `.env.example` ships with both keys blank, documenting the variables while
  keeping tracing off by default.
- Region is configurable, since Langfuse offers more than one hosted region.

## Alternatives considered

**Self-host the current Langfuse release.** Rejected on resources: roughly 3.5 GB
for ClickHouse, Redis, object storage, a worker and a second PostgreSQL, on a
machine that cannot spare it. This is the right answer on a larger machine, or
wherever query text must not leave the environment.

**Self-host an older, lighter Langfuse release.** Earlier versions ran as a
single container against one PostgreSQL database and would fit comfortably.
Rejected because it means building on a superseded version — inheriting a
migration problem in exchange for short-term convenience, and demonstrating a
deployment nobody would choose today.

**OpenTelemetry with Jaeger or Grafana Tempo.** A more standard and more portable
tracing stack, and genuinely tempting. Rejected because it provides spans but not
the LLM-specific layer that motivates this: no prompt versioning or comparison,
no token and cost accounting, and no mechanism for attaching evaluation scores or
user feedback to a specific generation. Those are the features Phase 4 depends
on, and rebuilding them over raw spans is a project in itself.

**Structured logs only.** The application already emits structured logs with a
request ID binding every stage of one request together, and this genuinely covers
part of the need. Rejected as sufficient: logs give no nested span view, no
side-by-side prompt version comparison, and no way to link a user's thumbs-down
to the exact retrieval and generation that produced it.

## References

- ADR-0006 — the RAM constraint that makes self-hosting infeasible here
- ADR-0003 — the "runnable without an account" principle this decision is in
  tension with
- Phase 4 — the observability and evaluation work this supports
