"""Getting papers from arXiv into Postgres and OpenSearch.

Sub-packages (Phase 1):

* ``arxiv_client``  -- async, rate-limited, retrying client for the arXiv API
* ``extraction``    -- PDF text extraction with an OCR fallback for scanned pages
* ``chunking``      -- structure-aware splitting that respects section boundaries
* ``embedding``     -- pluggable embedding providers and batched encoding
* ``indexing``      -- idempotent OpenSearch upserts driven by content hashes
* ``pipelines``     -- orchestration services callable from a CLI *or* an Airflow DAG

The pipeline services live here, not in the DAG files. An Airflow task should be a thin
call into this package, which keeps the pipeline testable without an Airflow scheduler.
"""

from __future__ import annotations
