# ADR-0001: Record architecture decisions

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-09-24 |
| **Deciders** | Pallavi Nile |
| **Supersedes** | — |
| **Superseded by** | — |

## Context

This project makes a number of decisions that are not obvious from reading the
code, and that a reasonable engineer could have decided differently:

- Two datastores (PostgreSQL *and* OpenSearch) rather than one.
- Retrieval that combines keyword scoring with vector similarity, rather than
  vectors alone as most tutorials do.
- A locally-served open-weights model rather than a hosted API.
- A `src/` package layout that departs from the structure originally sketched.

Source code records **what** a system does. It is a poor medium for **why**. The
usual places that "why" ends up all fail over time:

- **Code comments** drift. They are edited alongside the code they annotate, so
  the original reasoning is gradually overwritten by the current implementation.
- **Commit messages** are scattered. Reconstructing a single decision means
  finding the right commit among hundreds, and the decision often predates the
  commit that implements it.
- **Memory** fades. Six months after writing this, the tradeoffs that felt
  obvious will not be recallable under interview pressure.

That last point is not incidental. This repository is a portfolio artifact. Its
purpose is partly to demonstrate engineering judgement to people who will ask
questions like *"why not just use a vector database?"* A convincing answer
requires remembering not only what was chosen, but what was rejected and on what
grounds.

## Decision

Record every significant architectural decision as an Architecture Decision
Record (ADR): a short Markdown file in `docs/adr/`, numbered sequentially, using
the structure defined below.

An ADR is written when a decision meets any of these tests:

- It is costly or disruptive to reverse later (a datastore, an index schema, a
  package layout).
- A competent engineer could reasonably have chosen otherwise.
- It will provoke the question *"why did you do it that way?"*

Routine choices — a variable name, a helper function's signature, which HTTP
client library to use — do not warrant an ADR.

### Format

Each ADR uses these sections:

| Section | Contains |
|---|---|
| **Status** | `Proposed`, `Accepted`, `Deprecated`, or `Superseded by ADR-NNNN` |
| **Context** | The forces at play, stated *before* any solution. If a reader cannot see why the decision was hard, the context is incomplete. |
| **Decision** | What was chosen, in plain active voice: "We will…" |
| **Consequences** | What follows — **including the bad parts**. |
| **Alternatives considered** | What was rejected, and specifically why. |

### Immutability

**ADRs are not edited after acceptance.** When a decision changes, write a new
ADR and mark the old one `Superseded by ADR-NNNN`. The superseded record stays in
the repository.

This is the rule most often broken, and the one that carries the most value.
Editing an ADR to match current reality destroys exactly the information worth
keeping: that the team once believed something different, and what changed their
mind. A superseded ADR paired with its replacement is a complete argument. A
silently rewritten ADR is just documentation.

### Honest consequences

The **Consequences** section must record real drawbacks, not only benefits.

An ADR that lists only upsides is marketing, and an experienced reader discounts
it immediately — every genuine architectural decision costs something. Naming the
cost is what makes the rest of the document credible. Where this project's
measurements later contradict an ADR's expectations, the measurement wins and a
superseding ADR is written.

## Consequences

### Positive

- The reasoning behind each significant decision survives independently of the
  people who made it and of the code that implements it.
- Interview questions about architecture have pre-written, considered answers
  rather than improvised ones.
- New readers of the repository can understand intent without reverse-engineering
  it from the implementation.
- Writing the Context section before the Decision section surfaces weak reasoning
  early. A decision that cannot be justified in a page is usually not yet
  understood.

### Negative

- Writing ADRs takes time that could go into implementation.
- Superseded ADRs accumulate. A reader who finds ADR-0004 without noticing it is
  superseded may act on stale information — which is why status must appear at
  the very top of every file.
- The discipline degrades easily. A decision made in a hurry and never recorded
  leaves a gap that is rarely filled afterwards.

### Neutral

- ADR numbers are permanent and never reused, so gaps in the sequence are normal
  if a proposed ADR is abandoned.
- Records are plain Markdown in Git. No tooling, no database, no dependency —
  they are diffable, reviewable in pull requests, and readable on GitHub.

## Alternatives considered

**A single `ARCHITECTURE.md` document.** Rejected: one growing file has no way to
express that a decision was superseded. Editing it in place destroys history, and
its diffs become unreviewable as it grows.

**A project wiki or Notion page.** Rejected: documentation stored outside the
repository drifts from the code, is not reviewed alongside the change that
motivated it, and is invisible to anyone reading the project on GitHub. ADRs
travel with a `git clone`.

**Relying on commit messages alone.** Rejected: commit messages explain a change,
not a decision. One decision may span many commits, or precede all of them, and
there is no index.

**Recording nothing.** Rejected for the reason in Context: the reasoning would be
lost precisely when it is most needed.

## References

- Michael Nygard, *Documenting Architecture Decisions* (2011) — the original
  description of this practice, which this format follows.
