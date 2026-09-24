# ADR-0004: Combine BM25 and dense vector retrieval, and measure the result

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-09-24 |
| **Deciders** | Pallavi Nile |
| **Supersedes** | — |
| **Superseded by** | — |

## Context

Retrieval decides the ceiling on answer quality. A generated answer can only be
as good as the passages it was given; no prompt recovers from retrieving the
wrong chunks. The retrieval method is therefore the single most consequential
choice in this system.

Two established approaches exist, and they fail in different places.

### Lexical retrieval (BM25)

BM25 scores a document by the query terms it literally contains, weighted by how
rare each term is across the corpus, normalised for document length and with
term-frequency saturation so repetition has diminishing returns. It is the
default ranking function in Lucene, and therefore in OpenSearch.

**Where it fails: vocabulary mismatch.** A user asking *"how does a search system
decide which words matter most?"* shares almost no terms with a paper that says
*"inverse document frequency weighting"*. The paper is exactly on topic. BM25
scores it near zero, because the words do not overlap.

### Dense vector retrieval

An embedding model maps queries and chunks into a shared vector space where
proximity approximates semantic relatedness. The vocabulary-mismatch query above
works, because "which words matter most" and "inverse document frequency" land
near each other regardless of shared tokens.

**Where it fails: precise, rare, or out-of-distribution tokens.**

- **Identifiers.** `arXiv:2401.12345` has no meaningful embedding. Its vector is
  near other identifier-shaped strings, not near the paper it names.
- **Rare technical terms.** If the embedding model saw little about a specific
  method during training, a query naming it retrieves passages that are
  *generally about the area* rather than the passage that defines it. Asking
  *"what is BM25?"* can return chunks about ranking in general.
- **Exact phrases, acronyms, numbers, author surnames.** Embeddings compress;
  compression discards precision, and these are cases where precision is the
  whole request.

### The property that makes combining them worthwhile

Combining two methods only helps if **their failures are uncorrelated**. Two
methods that fail on the same queries produce an ensemble that also fails on
those queries.

Here the failure modes are close to complementary by construction. BM25 fails
when wording differs but meaning matches. Dense retrieval fails when wording
matches exactly but the token is rare or meaningless to the embedder. These are
close to opposite conditions — which is the actual argument for hybrid
retrieval, and a stronger one than "hybrid is best practice".

### What is not yet known

That argument is a **hypothesis**, not a result. It is plausible and widely
reported, but it has not been measured on *this* corpus with *this* embedding
model and *these* queries. On a small corpus of well-written academic abstracts,
queried by someone who already knows the terminology, BM25 alone may perform
comparably — and at lower latency and complexity.

## Decision

Implement **three interchangeable retrievers behind one interface** — BM25-only,
dense-only, and hybrid — and select between them by configuration.

Building all three is not indecision. It is what makes the comparison possible:
each is a legitimate production configuration, and switching between them for an
experiment is a config change rather than a rewrite.

### Fusion method: Reciprocal Rank Fusion by default

Hybrid retrieval must merge two result lists whose scores are not comparable.
BM25 scores are unbounded and depend on corpus statistics; cosine similarity is
bounded in `[-1, 1]`. Adding or averaging them directly is meaningless, and
min-max normalising them requires choosing a normalisation window that itself
biases the outcome.

**Reciprocal Rank Fusion sidesteps this by discarding scores and using only rank
position:**

```text
RRF(d) = Σ  1 / (k + rank_r(d))
         r
```

summed over each retriever `r` that returned document `d`, with `k = 60` by
convention. A document ranked highly by both retrievers accumulates
contributions from both; one ranked highly by a single retriever still surfaces.

RRF is the default because it requires no score normalisation and no per-corpus
tuning. **Weighted score fusion with min-max normalisation is also implemented**,
exposing tunable lexical and vector weights, so the choice of fusion method is
itself measurable rather than assumed.

### Supporting requirements

- Metadata filters (author, category, date range, paper ID) are applied **inside
  the search engine** during scoring, not to results afterwards. Post-filtering a
  top-k list silently returns fewer than k results.
- Overlapping chunks are deduplicated before fusion, so a passage appearing in
  two adjacent chunks does not occupy two slots.
- Each result carries **provenance**: which retrievers returned it, at what rank,
  with what score. Without this, a bad result cannot be diagnosed.

### The comparison is part of the decision

Phase 2 will measure four configurations — BM25 only, dense only, hybrid, and
hybrid plus re-ranking — against a hand-labelled evaluation set spanning exact
terminology, conceptual questions, method comparisons, paper-specific lookups,
multi-source questions, and deliberate vocabulary mismatches.

Metrics: Precision@K, Recall@K, MRR, nDCG@K, and latency for retrieval and
re-ranking separately.

**If hybrid does not win, the measured result is recorded and this ADR is
superseded.** No claim that hybrid retrieval is better will appear in the README,
the evaluation report, or any résumé bullet unless the recorded numbers support
it.

## Consequences

### Positive

- Query types that defeat either method alone remain answerable.
- Because all three retrievers are real configurations, the evaluation compares
  production code paths rather than experimental branches.
- RRF introduces no score-normalisation parameters, removing a class of tuning
  that is easy to get subtly wrong.
- Provenance makes retrieval explainable — for the debug panel in the UI, and
  for failure analysis in Phase 4.
- Both retrieval paths read the **same OpenSearch document** (ADR-0002), so there
  is one index to maintain and no cross-system join.

### Negative

- **Higher latency than either method alone.** Hybrid runs both retrieval paths
  and fuses the results. Dense retrieval also requires embedding the query at
  request time, which on a CPU-only machine is a non-trivial share of total
  latency. This will be measured, not estimated.
- **More parameters to get wrong**: the RRF constant, fusion weights, candidate
  pool size per retriever, final k. Each is a chance to tune on noise.
- **RRF discards score magnitude.** A document ranked first by a decisive margin
  is treated identically to one ranked first by a hair. Weighted score fusion
  preserves magnitude, which is part of why both are implemented.
- More code, more tests, more that can break, for a benefit not yet demonstrated
  on this corpus.
- Changing the embedding model invalidates every stored vector and forces a full
  reindex — cheap here only because OpenSearch is a derived index (ADR-0002).

### Neutral

- Re-ranking is a separate stage applied to the fused candidate pool, not part of
  fusion, and is configured independently.
- Fusion is implemented in application code rather than delegated to OpenSearch's
  built-in hybrid query pipeline. This keeps the method explicit, unit-testable
  against fixed rank lists with no cluster running, and swappable during
  experiments.

## Alternatives considered

**Dense vector retrieval alone.** The common tutorial approach, and the one most
RAG projects ship. Rejected because this corpus is academic papers, where users
legitimately search by exact identifier, method name, and author surname —
precisely the queries embeddings handle worst.

**BM25 alone.** Rejected as a default because conceptual questions phrased in the
user's own words are a primary use case, and that is exactly where lexical
matching fails. It remains implemented and will be measured; if it proves
competitive, that result stands.

**OpenSearch's built-in hybrid query with a normalisation search pipeline.** A
legitimate option that moves fusion into the engine. Not adopted: fusion in
application code can be unit-tested against synthetic rank lists without a
running cluster, swapped between RRF and weighted fusion for experiments, and
read directly by anyone reviewing the repository. Worth revisiting if
application-side fusion becomes a latency bottleneck.

**Query expansion or rewriting instead of dense retrieval** — using an LLM to
generate synonyms, then BM25 alone. Rejected as the primary mechanism: it adds an
LLM call to the critical path before retrieval even begins, and on CPU-only local
inference that cost is severe.

## References

- Cormack, Clarke & Büttcher (2009), *Reciprocal Rank Fusion outperforms Condorcet
  and individual Rank Learning Methods* — the origin of RRF and of `k = 60`.
- Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and
  Beyond*.
- ADR-0002 — why both retrieval paths read one OpenSearch document
