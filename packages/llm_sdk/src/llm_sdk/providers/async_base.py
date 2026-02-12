# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.domain.chat import ChatRequest, ChatResponse, ChatStreamEvent
from llm_sdk.domain.embeddings import EmbeddingRequest, EmbeddingResponse


class AsyncBaseLLMClient(ABC):
    """
    Async abstract base for LLM providers.

    Providers MUST:
    - raise ProviderError for failures
    - not leak httpx exceptions
    - implement aclose() if they own network resources
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Provider name used in registry.

        Returns:
            Provider name.
        """
        raise NotImplementedError


    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Perform async chat completion.

        Args:
            request: ChatRequest

        Returns:
            ChatResponse
        """
        raise NotImplementedError


    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Perform async embeddings.

        Args:
            request: EmbeddingRequest

        Returns:
            EmbeddingResponse
        """
        raise NotImplementedError


    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        """
        Stream chat tokens.

        Providers may override.

        Args:
            request: ChatRequest

        Returns:
            AsyncIterator[ChatStreamEvent]
        """
        raise NotImplementedError


    async def aclose(self) -> None:
        """
        Cleanup provider resources (e.g. AsyncClient).

        Returns:
            None
        """
        return None
