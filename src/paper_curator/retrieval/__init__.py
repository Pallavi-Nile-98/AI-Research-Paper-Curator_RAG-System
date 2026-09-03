"""Finding the right passages for a question.

Sub-packages (Phase 2):

* ``keyword``   -- BM25 lexical search over OpenSearch
* ``vector``    -- dense k-NN search over the same index
* ``fusion``    -- combining two ranked lists (Reciprocal Rank Fusion / weighted scores)
* ``reranking`` -- cross-encoder re-scoring of a wide candidate pool
* ``context``   -- assembling the final prompt context with stable citation IDs

Every retriever implements the same interface, so BM25-only, vector-only, hybrid, and
hybrid-plus-re-ranking can be swapped by configuration. That is what makes the Phase 2
comparison experiments an honest apples-to-apples measurement.
"""

from __future__ import annotations
