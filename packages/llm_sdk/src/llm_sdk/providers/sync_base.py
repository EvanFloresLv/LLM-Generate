# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from abc import ABC, abstractmethod

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.domain.chat import ChatRequest, ChatResponse, ChatStream
from llm_sdk.domain.embeddings import EmbeddingRequest, EmbeddingResponse


class BaseLLMClient(ABC):
    """
    Abstract base for all LLM providers.

    Providers MUST:
    - raise ProviderError for provider failures
    - not leak raw httpx exceptions
    - be thread-safe or document if not
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
    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Perform chat completion.

        Args:
            request: ChatRequest

        Returns:
            ChatResponse
        """
        raise NotImplementedError


    @abstractmethod
    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Perform embeddings.

        Args:
            request: EmbeddingRequest

        Returns:
            EmbeddingResponse
        """
        raise NotImplementedError


    def stream_chat(self, request: ChatRequest) -> ChatStream:
        """
        Stream chat tokens.

        Providers may override. Default raises NotImplementedError.

        Args:
            request: ChatRequest

        Returns:
            Iterator of ChatStreamEvent
        """
        raise NotImplementedError
