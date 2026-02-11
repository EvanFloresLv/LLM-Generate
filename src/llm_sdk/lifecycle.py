# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.providers.async_base import AsyncBaseLLMClient


@dataclass(frozen=True)
class AsyncResourceManager:
    """
    Tracks async provider clients and closes them on shutdown.

    This prevents:
    - leaking open connections
    - creating new AsyncClient per request

    Note:
        Providers created via registry factories are cached by provider name.
    """

    _clients: dict[str, AsyncBaseLLMClient] = field(default_factory=dict)

    def get_cached(self, provider: str) -> AsyncBaseLLMClient | None:
        """
        Get cached client.

        Args:
            provider: Provider name.

        Returns:
            AsyncBaseLLMClient | None
        """
        return self._clients.get(provider)


    def set_cached(self, provider: str, client: AsyncBaseLLMClient) -> None:
        """
        Create and cache client.

        Args:
            provider: Provider name.
        """
        self._clients[provider] = client


    async def aclose(self) -> None:
        """
        Close all cached clients.

        Returns:
            None
        """
        for c in self._clients.values():
            await c.aclose()

        self._clients.clear()


@asynccontextmanager
async def lifespan(manager: AsyncResourceManager) -> AsyncIterator[AsyncResourceManager]:
    """
    Async context manager for SDK lifecycle.

    Args:
        manager: AsyncResourceManager

    Yields:
        manager
    """
    try:
        yield manager
    finally:
        await manager.aclose()