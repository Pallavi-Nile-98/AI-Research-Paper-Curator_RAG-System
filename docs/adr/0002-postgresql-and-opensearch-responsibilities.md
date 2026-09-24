# ADR-0002: Split responsibilities between PostgreSQL and OpenSearch

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-09-24 |
| **Deciders** | Pallavi Nile |
| **Supersedes** | — |
| **Superseded by** | — |

## Context

This system stores data for two purposes that pull in opposite directions.

**Pipeline state must be correct.** Ingestion is a multi-step process — fetch
metadata, download a PDF, extract text, chunk it, embed the chunks, index them —
and any step can fail. The system must know, reliably: which papers have been
seen, at which version, which are still being processed, which failed and how
many times, and where the last incremental sync stopped. Re-running the pipeline
must not create duplicates. A user's feedback on an answer must not be lost.

**Retrieval must be fast and relevant.** Answering a question means scoring
thousands of text chunks by both lexical overlap and semantic similarity, in
well under a second, with filters applied.

The first set of requirements is what relational databases are built for:
transactions, foreign keys, unique constraints, and read-your-writes
consistency. The second is what search engines are built for: inverted indexes,
BM25 scoring with proper text analysis, and approximate nearest-neighbour vector
search.

Using one system for both means accepting whichever weakness that system has.

### Why one store is genuinely tempting

This deserves honesty rather than a straw man. **PostgreSQL alone could do this
job at this scale.** It has built-in full-text search via `tsvector`, and the
`pgvector` extension provides HNSW-indexed vector similarity that performs well
into the millions of rows. For a corpus of a few thousand papers, a single
Postgres instance would work, and would be simpler to operate.

The reasons not to are narrower than "Postgres can't search":

- **Ranking quality.** Postgres full-text ranking (`ts_rank`, `ts_rank_cd`) is
  frequency-based. It does not implement BM25, and lacks BM25's document-length
  normalisation and term-frequency saturation — the properties that stop long
  documents and repeated terms from dominating results. Lucene's BM25 is the
  reference implementation, with mature analyzers for stemming and stopwords.
- **Hybrid ergonomics.** Combining lexical and vector scores in Postgres means
  two separate queries and fusion written by hand in SQL. In OpenSearch a chunk
  is one document carrying both its text and its vector, so both retrieval paths
  read the same document and fusion happens against one result set.
- **Production parity.** The deployment target is Amazon OpenSearch Service.
  Developing against the same engine avoids discovering behavioural differences
  in relevance tuning at deployment time.

### Why OpenSearch alone is worse

Storing pipeline state in OpenSearch fails on properties that are not
negotiable:

- **No transactions.** If a pipeline run dies after writing a paper record but
  before writing its processing status, there is no rollback. The next run reads
  inconsistent state.
- **No unique constraints across documents.** Idempotent ingestion depends on a
  unique constraint over `(arxiv_id, version)`. Deterministic document IDs can
  approximate this for a single document, but cannot express constraints
  spanning documents, and cannot express foreign keys at all.
- **Writes are not immediately readable.** OpenSearch is near-real-time: an
  indexed document becomes searchable after the next refresh, one second by
  default. Pipeline state requires read-your-writes — a checkpoint written and
  then immediately read must return the value just written.
- **No schema migrations.** Changing an index mapping generally means reindexing.
  That is acceptable for derived data and unacceptable for the only copy of user
  feedback.

## Decision

**PostgreSQL is the system of record. OpenSearch is a derived index.**

PostgreSQL owns:

| Data | Why it belongs here |
|---|---|
| Papers, authors, paper-author relationships | Foreign keys; a unique constraint on `(arxiv_id, version)` |
| Ingestion runs and pipeline checkpoints | Transactional; must be read-your-writes |
| Per-document processing status and retry counts | Mutable state with a defined lifecycle |
| Failed-document records (dead letter) | Must survive and be queryable relationally |
| User queries and feedback | User-generated; cannot be regenerated if lost |
| Chunk metadata and content hashes | Drives synchronisation with the index |

OpenSearch owns:

| Data | Why it belongs here |
|---|---|
| Chunk text, analysed for BM25 | Inverted index with real analyzers |
| Chunk embeddings as `knn_vector` | HNSW approximate nearest-neighbour search |
| Chunk metadata copied for filtering | Filters must apply during scoring, not after |

### The consequence that matters most

**OpenSearch holds nothing that cannot be rebuilt from PostgreSQL.**

Every indexed document is derived from rows Postgres already has. If the search
cluster is lost, corrupted, or needs a different mapping, recovery is a reindex
from Postgres — not data loss. This is what makes index versioning with alias
swaps safe, and it is why changing the embedding model is an operational task
rather than a crisis.

Stated as a rule: **anything that cannot be regenerated lives in PostgreSQL.**

## Consequences

### Positive

- Each store is used for what it is actually good at.
- Ingestion is idempotent by database constraint rather than by application
  logic that has to be right every time.
- Alembic gives schema changes that are versioned, reviewable in pull requests,
  and reversible.
- Index rebuilds and mapping changes are routine, because the source of truth is
  elsewhere.
- The split maps directly onto managed AWS services — Amazon RDS for PostgreSQL
  and Amazon OpenSearch Service — with no architectural change at deployment.

### Negative

- **Two systems to run, learn, monitor and pay for.** This is a real cost and the
  main argument against the decision. It is why the "Postgres alone" alternative
  is recorded above as viable rather than wrong.
- **Dual-write consistency is now the system's hardest problem.** Postgres can
  commit that a chunk exists while the OpenSearch write fails, leaving the index
  incomplete but the state table claiming success. There is no distributed
  transaction here. It is handled by making indexing idempotent with
  deterministic document IDs, marking chunks indexed only after the bulk write is
  acknowledged, and reconciling by content hash — but it requires deliberate work
  in Phase 1 and it is where bugs are most likely.
- Two query languages: SQL for state, OpenSearch DSL for retrieval.
- Roughly 2.3 GB of local RAM for both, which is what makes Compose profiles
  necessary (ADR-0006).
- Higher AWS cost than a single RDS instance.

### Neutral

- Chunk metadata is deliberately duplicated into OpenSearch. Normalisation is a
  property of the system of record; a search index is denormalised on purpose so
  filters can be applied during scoring.
- Postgres retains content hashes for every chunk, so drift between the two
  stores is detectable rather than merely suspected.

## Alternatives considered

**PostgreSQL alone, with `tsvector` and `pgvector`.** The strongest alternative,
and workable at this corpus size. Rejected for the three reasons in Context:
`ts_rank` is not BM25, hybrid fusion would be hand-written SQL over two separate
queries, and it diverges from the deployment target. Had operational simplicity
been the overriding goal, this would have been the right answer.

**OpenSearch alone.** Rejected on correctness, not performance: no transactions,
no cross-document unique constraints, near-real-time refresh breaking
read-your-writes for checkpoints, and no migration story for user feedback.

**PostgreSQL plus a dedicated vector database** (Pinecone, Weaviate, Qdrant).
Rejected: a third system to operate, and separating lexical from vector
retrieval means two network round trips and client-side fusion over two
independently-scored result sets — the exact ergonomic problem that motivated
keeping text and vector in one document.

**Elasticsearch instead of OpenSearch.** Rejected primarily for deployment
alignment: AWS's managed offering is Amazon OpenSearch Service, and OpenSearch is
the Apache-2.0-licensed fork created after Elastic's 2021 licence change. For a
public portfolio repository, a permissive licence and a matching managed service
both matter. The two are close enough technically that the retrieval design here
would transfer.

## References

- ADR-0004 — Hybrid retrieval combining BM25 and dense vectors
- ADR-0006 — Compose profiles, which exist partly because this decision needs two services
- [OpenSearch k-NN documentation](https://docs.opensearch.org/latest/vector-search/)
- [PostgreSQL full-text search ranking](https://www.postgresql.org/docs/current/textsearch-controls.html)
