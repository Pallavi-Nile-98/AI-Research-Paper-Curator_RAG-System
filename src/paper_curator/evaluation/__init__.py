"""Measuring retrieval and answer quality (Phases 2 and 4).

Contains the *code*: metric implementations (Precision@K, Recall@K, MRR, nDCG@K),
experiment runners, and the RAGAS harness.

The *artefacts* -- labelled datasets, experiment configs and result reports -- live in
the top-level ``evaluation/`` directory instead, where they are visible to anyone
browsing the repository. Those files are the evidence behind any number that ends up on
a resume, so they are version-controlled deliberately.
"""

from __future__ import annotations
