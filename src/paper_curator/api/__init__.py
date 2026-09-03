"""FastAPI application: HTTP surface only.

Sub-packages (Phase 3):

* ``routes``     -- thin handlers; parse, delegate, serialise
* ``schemas``    -- Pydantic request/response contracts
* ``services``   -- use-case orchestration wiring retrieval and generation together
* ``middleware`` -- request IDs, structured access logs, error mapping, size limits

Route handlers contain no business logic. A handler that grew a ranking heuristic would
make that heuristic untestable without an HTTP client, and unusable from the CLI or an
Airflow task.
"""

from __future__ import annotations
