"""Tracing and instrumentation (Phase 4).

Wraps Langfuse behind a small interface with a no-op implementation. When credentials
are absent the no-op is used and the application behaves identically, so observability
can never become a runtime dependency of answering a question.

Traces are sent to Langfuse Cloud, a third-party service. The tracing layer is therefore
explicitly responsible for *not* attaching secrets or unnecessary personal data.
"""

from __future__ import annotations
