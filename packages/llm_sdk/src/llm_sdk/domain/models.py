# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Usage:
    """
    Token usage info (provider normalized).
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    thought_tokens: int | None = None
    total_tokens: int | None = None