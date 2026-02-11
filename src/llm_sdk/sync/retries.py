# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TypeVar

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.exceptions import ProviderError, TimeoutError


T = TypeVar("T")

@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """
    Retry policy with exponential backoff + jitter.

    Args:
        max_attempts: Total attempts (1 means no retries).
        base_delay_s: Initial delay.
        max_delay_s: Max delay cap.
    """

    max_attempts: int = 3
    base_delay_s: float = 0.25
    max_delay_s: float = 3.0


    def compute_delay(self, attempt: int) -> float:
        """
        Compute the delay for a specific retry attempt.

        Args:
            attempt: The current retry attempt (0-indexed).

        Returns:
            The computed delay in seconds.
        """
        exp = min(self.max_delay_s, self.base_delay_s * (2 ** attempt - 1))
        jitter = random.uniform(0, self.base_delay_s)
        return exp + jitter


def with_retries(
    *,
    fn: callable[[], T],
    provider: str,
    retry_policy: RetryPolicy
):
    """
    Execute a function with retry policy.

    Retries only ProviderError(is_retryable=True).
    Converts repeated retryable errors into TimeoutError.

    Args:
        fn: Callable without args.
        provider: Provider name.
        policy: RetryPolicy.

    Returns:
        Result of fn.

    Raises:
        ProviderError: If non-retryable provider error occurs.
        TimeoutError: If retries exhausted.
    """

    last_error: ProviderError | None = None

    for attempt in range(retry_policy.max_attempts):
        try:
            return fn()
        except ProviderError as e:
            if not e.is_retryable:
                raise
            last_error = e
            time.sleep(retry_policy.compute_delay(attempt))

    raise TimeoutError(provider=provider, last_error=last_error)