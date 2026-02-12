# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass


class LLMError(Exception):
    """Base exception for LLM-related errors."""


@dataclass(slots=True)
class ProviderError(LLMError):
    """
    Raised when a provider fails.

    Args:
        provider: Provider name.
        message: Human readable message.
        status_code: Optional HTTP status.
        is_retryable: Whether core should retry.
    """

    provider: str
    message: str
    status_code: int | None = None
    is_retryable: bool = False

    def __str__(self):
        return f"ProviderError(provider={self.provider}, message={self.message}, status_code={self.status_code}, is_retryable={self.is_retryable})"


class ValidationError(LLMError):
    """Raised when a request is invalid."""


class ProviderNotFoundError(LLMError):
    """Raised when provider is not registered."""


class ModelNotFoundError(LLMError):
    """Raised when model is not found in registry."""


class StreamingNotSupportedError(LLMError):
    """Raised when a provider does not support streaming."""


@dataclass(frozen=True, slots=True)
class TimeoutError(LLMError):
    """
    Raised when a request times out after exhausting retries.

    Args:
        provider: Provider name.
        last_error: Last exception raised before timeout.
    """

    provider: str
    last_error: Exception | None = None

    def __str__(self) -> str:
        return f"LLMTimeoutError(provider={self.provider}, last_error={self.last_error})"