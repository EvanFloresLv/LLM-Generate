# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeoutConfig:
    """
    Timeout configuration.

    Args:
        connect: Connect timeout (seconds).
        read: Read timeout (seconds).
        write: Write timeout (seconds).
        pool: Pool timeout (seconds).
    """

    connect: float = 5.0
    read: float = 60.0
    write: float = 60.0
    pool: float = 5.0
