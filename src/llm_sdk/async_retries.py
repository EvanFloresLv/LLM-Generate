# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.exceptions import ProviderError, TimeoutError

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AsyncRetryPolicy:
    """
    Async retry policy with exponential backoff + jitter.

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
        Compute backoff delay.

        Args:
            attempt: Attempt number starting at 1.

        Returns:
            Delay in seconds.
        """
        exp = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        jitter = random.uniform(0, exp)
        return jitter


async def with_async_retries(
    *,
    fn: Callable[[], Awaitable[T]],
    provider: str,
    policy: AsyncRetryPolicy
) -> T:
    """
    Execute an async function with retry policy.

    Retries only ProviderError(is_retryable=True).

    Args:
        fn: Async callable without args.
        provider: Provider name.
        policy: AsyncRetryPolicy.

    Returns:
        Result of fn.

    Raises:
        ProviderError: If non-retryable error.
        TimeoutError: If retries exhausted.
    """

    last_error: ProviderError | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await fn()
        except ProviderError as e:
            last_error = e
            if not e.is_retryable or attempt >= policy.max_attempts:
                raise
            await asyncio.sleep(policy.compute_delay(attempt))

    raise TimeoutError(f"[{provider}] retries exhausted: {last_error}")
