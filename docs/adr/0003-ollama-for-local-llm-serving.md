# ADR-0003: Serve the generation model locally with Ollama

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-09-24 |
| **Deciders** | Pallavi Nile |
| **Supersedes** | — |
| **Superseded by** | — |

## Context

The final stage of the pipeline gives retrieved passages to a language model and
asks for an answer that cites them. Three ways to obtain that model:

1. A hosted API (OpenAI, Anthropic, and similar).
2. A managed cloud model service (Amazon Bedrock).
3. An open-weights model served locally.

The obvious choice on **answer quality alone is a hosted frontier model**. It
would follow citation instructions more reliably, produce valid structured output
more often, and reason better over conflicting sources. That is not in dispute,
and the decision below accepts a real quality penalty.

Three considerations pull the other way.

### Evaluation cost scales with rigour

This project's value rests on measured results, not claims. Phase 4 compares
retrieval with and without re-ranking, and at least one prompt variation, across
a held-out question set, with RAGAS computing faithfulness, answer relevancy and
context precision — metrics that themselves invoke a model per sample.

That is configurations × questions × metric calls, repeated whenever a prompt or
retrieval parameter changes. With per-token billing, the marginal cost of running
an evaluation again discourages running it again. **The failure mode is not a
large bill; it is quietly evaluating less thoroughly to avoid one.** Local
inference makes the marginal cost of a re-run approximately zero, which is what
keeps the measurement honest.

### Recorded results must stay reproducible

Evaluation numbers will appear in a public repository and on a résumé, and must
be defensible months later. Hosted models are moving targets: versions are
updated and retired, and identical prompts return different output over time.
A pinned local model at fixed temperature is far more stable, so a number
recorded today can be regenerated later.

### The repository must run without an account

This is a public portfolio project. Someone cloning it should be able to run the
system end to end without creating an account, entering a credit card, or
obtaining an API key. A hosted dependency in the critical path makes the project
unrunnable for exactly the audience it is meant to demonstrate to.

Secondary: queries and paper content never leave the machine.

## Decision

Serve generation with **Ollama**, running as a Docker Compose service under the
`llm` profile (ADR-0006), with a configurable model defaulting to a small
instruction-tuned model suited to CPU-only inference.

**The load-bearing part of this decision is the interface, not Ollama.**

Generation sits behind a narrow client abstraction: given a prompt and
parameters, return text plus token counts and latency. Ollama is one
implementation. A hosted-API implementation is a second. Tests use a fake that
returns canned responses with no model running at all.

This matters because the reasons above are specific to *this stage* of the
project. Evaluation cost dominates while measurements are being established; it
stops dominating once they are recorded. Answer quality then dominates. The
abstraction means that shift is a configuration change and one adapter class —
not a rewrite of the generation layer.

Ollama specifically, rather than another local runner, because it manages model
download, quantisation and memory lifecycle, and exposes them over **HTTP**. That
HTTP boundary matters: it is the same shape as a hosted API, so the two
implementations differ in endpoint and payload rather than in architecture.

Required client behaviour: async, explicit request timeouts, bounded retries with
backoff, a distinct error for "model not available" versus "request failed",
token and latency logging, and structured-output validation with a clear failure
path when the model returns something unparseable.

## Consequences

### Positive

- Running an evaluation again is free, so evaluations get run again.
- Recorded results are reproducible, because the model is pinned rather than
  managed by someone else.
- The repository is runnable by anyone who clones it, with no account and no key.
- Queries and paper content stay on the machine.
- Operating a self-hosted model server — quantisation, memory limits, timeout and
  failure handling — is a demonstrable skill in its own right, and one that a
  hosted API hides entirely.

### Negative

- **Answer quality will be visibly lower than a frontier hosted model.** This is
  the cost, stated plainly. A small model follows citation formats less
  reliably, produces invalid structured output more often, and handles
  conflicting sources less well. Some of what the evaluation measures will be
  model capability rather than retrieval quality, and the report must separate
  the two rather than attributing every failure to retrieval.
- **CPU-only inference is slow.** This machine has no CUDA GPU, so expect
  single-digit tokens per second. End-to-end latency will be dominated by
  generation, not retrieval — so retrieval and generation latency must be
  reported separately, or the retrieval numbers become meaningless.
- Roughly 3 GB of RAM while a model is loaded, which is why Ollama sits behind a
  profile rather than starting by default.
- One more service to run, monitor and fail gracefully around.
- **Model licensing needs review before the README recommends a default.** The
  intended default ships under a community licence with usage restrictions rather
  than an OSI-approved open-source licence. Since this repository is already
  public, that review is scheduled earlier than originally planned, and the
  outcome may change which model is recommended.

### Neutral

- Model weights live in a named Docker volume, so `docker-compose down` does not
  trigger a multi-gigabyte re-download.
- Temperature defaults low: this is grounded question answering over supplied
  evidence, not open-ended writing.
- The model is unloaded after a short idle period, so a forgotten container does
  not hold several gigabytes indefinitely.

## Alternatives considered

**A hosted frontier API as the default.** The best choice for answer quality, and
the natural upgrade once measurements are established. Rejected for now on three
grounds: per-token cost during evaluation discourages thorough re-running,
hosted model versions drift so recorded results stop being reproducible, and an
API key requirement makes a public portfolio repository unrunnable for anyone
who just wants to try it. The client abstraction exists precisely so this can be
revisited without rework.

**Amazon Bedrock.** Fits the AWS deployment architecture and removes the model
from the local resource budget. Rejected as the default because it requires AWS
credentials to run locally — reintroducing the account problem — and because
inference charges accrue during exactly the phase where evaluation volume is
highest. Recorded in Phase 5 as the recommended managed option for a deployed
version.

**Loading the model in-process with `transformers`.** Rejected: model memory
would be tied to the API process lifecycle, so a worker restart re-loads several
gigabytes, and scaling API workers would multiply model copies. It also removes
the HTTP boundary that makes the hosted-API swap straightforward.

**Running `llama.cpp` or vLLM directly.** vLLM targets GPU serving and would not
help on this machine. `llama.cpp` is essentially what Ollama wraps; using it
directly means implementing model management, quantisation selection and an HTTP
layer already provided.

## References

- ADR-0006 — Compose profiles, which is why Ollama does not start by default
- Phase 4 — RAGAS evaluation, whose cost profile drives this decision
- Phase 5 — AWS deployment, where the managed-model alternative is revisited
