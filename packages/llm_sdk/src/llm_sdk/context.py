# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Context:
    """
    Structured log context.

    Args:
        provider: Provider name.
        model: Model name.
        request_id: Optional request id.
    """

    provider: str
    model: str
    request_id: str | None = None
