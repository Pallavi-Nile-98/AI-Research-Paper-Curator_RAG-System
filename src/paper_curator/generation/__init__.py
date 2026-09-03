"""Turning retrieved context into a grounded, cited answer.

Contains (Phase 3):

* ``prompts/``  -- version-controlled prompt templates shipped as package data
* ``ollama/``   -- async LLM client with timeouts, bounded retries and a fake for tests
* citation parsing and validation

The prompt is treated as a versioned artefact, not a string literal. Every traced
generation records which prompt version produced it, so a quality regression can be
attributed to a specific change.
"""

from __future__ import annotations
