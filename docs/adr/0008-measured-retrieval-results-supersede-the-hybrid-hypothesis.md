# ADR-0008: Measured retrieval results supersede the hybrid hypothesis

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-10-04 |
| **Deciders** | Pallavi Nile |
| **Supersedes** | [ADR-0004](0004-hybrid-retrieval-with-bm25-and-dense-vectors.md) (its conclusion, not its reasoning) |
| **Superseded by** | — |

## Context

ADR-0004 argued that hybrid retrieval should outperform either BM25 or dense
retrieval alone, because their failure modes are uncorrelated. It recorded that
argument explicitly as a **hypothesis**, and committed to this:

> If hybrid does not win, the measured result is recorded and this ADR is
> superseded. No claim that hybrid retrieval is better will appear in the
> README, the evaluation report, or any résumé bullet unless the recorded
> numbers support it.

The experiment has now run. Hybrid did not win.

### What was measured

Thirty labelled queries against a corpus of 124 arXiv papers, indexed as 4,931
chunks. Four configurations, evaluated with paper-level relevance. Full
per-query results in `evaluation/experiments/retrieval_experiments.json`.

| configuration | recall@5 | recall@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| bm25 | 0.722 | 0.767 | 0.702 | 0.713 |
| **dense** | **0.828** | **0.911** | **0.815** | **0.839** |
| hybrid | 0.811 | 0.867 | 0.799 | 0.834 |
| hybrid + re-ranking | 0.772 | 0.883 | 0.791 | 0.828 |

Dense retrieval alone leads on every headline metric. Re-ranking made results
slightly worse while costing a median of 5,068 ms.

### The mechanism in ADR-0004 was confirmed

The reasoning was not wrong about *why* the two retrievers differ. Recall@10 by
category:

| category | bm25 | dense |
|---|---|---|
| exact_terminology | 1.000 | 1.000 |
| conceptual | 0.857 | 1.000 |
| **vocabulary_mismatch** | **0.400** | **0.800** |
| multi_source | 0.800 | 0.767 |
| paper_specific | 0.750 | 1.000 |

BM25 collapses on vocabulary mismatch, exactly as predicted. Three of its four
complete misses are vocabulary-mismatch queries. The failure mode is real and
measurable.

What did not hold is the inference drawn from it. Dense retrieval covered
BM25's weaknesses **without needing BM25**: it had zero complete misses across
all thirty queries. The predicted BM25 advantage on exact terminology did not
materialise either -- both retrievers scored 1.000, so that category
discriminates nothing on this corpus.

### Fusion actively destroyed a result

Query `q026` ("Why do chatbots sometimes make things up?") was found by dense
retrieval and **lost by hybrid**. Reciprocal rank fusion gives each retriever
equal weight by construction, so when one is substantially stronger, the
weaker one's rankings dilute it rather than complementing it. The ensemble
argument assumed two roughly comparable signals; here they were not comparable.

### A confound found while analysing the results

Query `q020` is a bare arXiv identifier, predicted to be BM25's clearest win.
BM25 did not retrieve the paper at all, while dense found it at rank 7.

The cause is a defect, not a property of BM25: the keyword retriever searches
`text`, `text.exact` and `title`, and never `arxiv_id`. That field exists in
the mapping only as a filter. A user pasting an identifier cannot find the
paper lexically.

This is being fixed separately. It affected one query of thirty and cannot
account for the overall gap, but it means BM25's `paper_specific` figure is an
underestimate.

## Decision

**Record the measured result. Supersede ADR-0004's conclusion. Do not change
the production default yet.**

Concretely:

1. ADR-0004 is marked superseded. Its *reasoning about failure modes* stands
   and is confirmed by the data; its *conclusion that hybrid wins* does not.
2. No claim of hybrid superiority appears in the README, the evaluation report,
   or any résumé bullet. What may be claimed is what was measured.
3. All three retrievers remain. Deleting working code on the strength of one
   provisional experiment would be the same mistake as adopting hybrid without
   measuring it.
4. The default retrieval mode **stays hybrid for now**, which deserves
   explaining, because the measurement says dense.

### Why the default does not change yet

The evidence favouring dense is real but comes from a dataset with a serious
threat to validity: **the questions were drafted by the same process that
built the system**, as paraphrases written from paper abstracts. Paraphrase is
precisely what dense retrieval is good at. The dataset may therefore be
measuring its own construction rather than retrieval quality, and the thirty
labels have not yet been reviewed by a human.

Switching a production default on the strength of a benchmark one may have
inadvertently shaped in one's own favour is worse practice than not measuring
at all, because it carries the appearance of rigour. The honest position is
that the result is **provisional**, and provisional results do not change
defaults.

### What would settle it

The default changes to dense if, after the labels are human-reviewed and the
`arxiv_id` defect is fixed, a re-run still shows dense ahead on nDCG@10 and
recall@10. The default stays hybrid if the gap closes, or if review shows the
questions were systematically biased toward paraphrase.

## Consequences

### Positive

- The project now has measured retrieval numbers rather than assumed ones, and
  a recorded procedure for reproducing them.
- A genuine defect in the keyword retriever was found by evaluation rather than
  by a user.
- The limits of the result are documented alongside it, so nobody -- including
  its author six months from now -- can mistake a provisional figure for a
  settled one.
- Keeping the measurement that contradicts the original design argument is the
  practice ADR-0001 describes. A superseded record paired with its replacement
  is a complete argument; quietly deleting the prediction would leave only a
  tidier and less honest story.

### Negative

- Hybrid retrieval costs roughly 4.9× BM25's median latency (227 ms against
  46 ms) and currently earns no measured quality benefit on this corpus. That
  cost is being paid for a hypothesis the data does not yet support.
- Re-ranking adds a median 5,068 ms and lowered nDCG@10 from 0.815 to 0.791.
  On current evidence it is a cost with no return. It stays implemented,
  disabled by default, because the result is provisional and because the
  component is correct even if the benefit is not demonstrated.
- The default now differs from what the measurement favours, which requires
  this explanation every time it is questioned.

### Neutral

- The corpus is 124 papers from a narrow slice of cs.IR. Relative performance
  at this scale may not hold at 10,000 papers, where lexical precision usually
  matters more because there are far more semantically-near distractors.
- Labels are paper-level and drawn from abstracts, so a paper that answers a
  question only in its body is scored as a miss for every configuration
  equally.

## Alternatives considered

**Switch the default to dense immediately.** Rejected for now: the dataset's
construction plausibly favours dense, and the labels are unreviewed. Revisit
after review.

**Keep ADR-0004 as accepted and record the result only in the report.**
Rejected: ADR-0004 states a conclusion the data does not support, and leaving
it unmarked means a future reader acts on a superseded belief. ADR-0001 exists
to prevent exactly that.

**Remove hybrid retrieval and the re-ranker.** Rejected: one provisional
experiment on a small corpus is not grounds for deleting correct, tested code,
and the comparison cannot be repeated after the alternatives are gone.

**Re-run after fixing the `arxiv_id` defect, before recording anything.**
Rejected as a precondition: the defect affects one query of thirty and cannot
explain the overall gap. Recording what was measured, then fixing and
re-measuring, keeps both results rather than replacing an inconvenient one.

## References

- `evaluation/datasets/retrieval_eval_v0.1-draft.json` — the labelled queries
- `evaluation/experiments/retrieval_experiments.json` — every per-query result
  and the full configuration of each run
- `evaluation/reports/retrieval_evaluation.md` — the generated report
- [ADR-0001](0001-record-architecture-decisions.md) — why a superseded record is kept rather than edited
- [ADR-0004](0004-hybrid-retrieval-with-bm25-and-dense-vectors.md) — the hypothesis this supersedes
