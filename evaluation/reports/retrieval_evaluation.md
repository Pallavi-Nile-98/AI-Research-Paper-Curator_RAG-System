# Retrieval evaluation

Generated: 2026-10-05T00:56:29.946180+00:00

## Dataset status

- Version: `0.1.0-draft`
- 30 examples, 0 human-reviewed, 30 UNREVIEWED DRAFT -- results computed against unreviewed labels are provisional
- Corpus: 124 arXiv papers, predominantly cs.IR, mostly submitted January 2024, indexed as 4,931 chunks. Questions were written against this specific corpus and are not portable to another one.

> **These results are provisional.** The relevance labels have not been
> reviewed by a human. Every number below is measured correctly against
> labels that may themselves be wrong, so no conclusion here should be
> quoted as an established result.

## Overall

| configuration | recall@5 | recall@10 | ndcg@10 | mrr | hit_rate@5 |
|---|---|---|---|---|---|
| bm25 | 0.722 | 0.767 | 0.702 | 0.713 | 0.800 |
| dense | 0.828 | 0.911 | 0.815 | 0.839 | 0.933 |
| hybrid | 0.811 | 0.867 | 0.799 | 0.834 | 0.933 |
| hybrid_rerank | 0.772 | 0.883 | 0.791 | 0.828 | 0.867 |

## Latency

| configuration | median ms | p90 ms | max ms | rerank median ms |
|---|---|---|---|---|
| bm25 | 46 | 96 | 2871 | - |
| dense | 72 | 127 | 1162 | - |
| hybrid | 227 | 322 | 327 | - |
| hybrid_rerank | 440 | 550 | 827 | 5068 |

## Recall@10 by query category

| category | bm25 | dense | hybrid | hybrid_rerank |
|---|---|---|---|---|
| conceptual | 0.857 | 1.000 | 0.929 | 1.000 |
| exact_terminology | 1.000 | 1.000 | 1.000 | 1.000 |
| method_comparison | 0.750 | 0.875 | 0.875 | 0.875 |
| multi_source | 0.800 | 0.767 | 0.800 | 0.800 |
| paper_specific | 0.750 | 1.000 | 0.750 | 0.750 |
| vocabulary_mismatch | 0.400 | 0.800 | 0.800 | 0.800 |

## Complete misses

Queries where a configuration retrieved no relevant paper at all.

**bm25** — 4 shown
- `q020` (paper_specific): arXiv 2401.05939
- `q026` (vocabulary_mismatch): Why do chatbots sometimes make things up?
- `q029` (vocabulary_mismatch): Teaching software to put search results in a better order by observing what people click
- `q030` (vocabulary_mismatch): Somebody deliberately feeding false information into a shopping site's suggestion engine

**dense** — 0 shown
- none

**hybrid** — 1 shown
- `q026` (vocabulary_mismatch): Why do chatbots sometimes make things up?

**hybrid_rerank** — 1 shown
- `q026` (vocabulary_mismatch): Why do chatbots sometimes make things up?

## Interpretation

Measured facts are above. The following is interpretation and is labelled
as such.

- Highest nDCG@10: **dense** at 0.815.
- The corpus is small and drawn from a narrow slice of cs.IR, so these
  figures describe this corpus and do not generalise.
- Labels are paper-level, taken from abstracts. A paper answering a question
  only in its body would be scored as a miss.

## Reproducing

```bash
docker-compose up -d
python scripts/evaluate_retrieval.py
```

The full configuration of every run, including model names and fusion
parameters, is recorded in `evaluation\experiments\retrieval_experiments.json`.
