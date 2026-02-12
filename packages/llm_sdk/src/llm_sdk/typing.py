# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


dataclass(frozen=True, slots=True)
class Result(Generic[T]):
    """
    Small Result type to avoid leaking provider exceptions.

    Attributes:
        ok: True if success.
        value: Result value (when ok=True).
        error: Error message (when ok=False).
    """

    ok: bool
    value: T | None = None
    error: str | None = None