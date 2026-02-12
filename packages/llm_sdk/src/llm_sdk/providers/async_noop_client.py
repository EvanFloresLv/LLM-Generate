# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import AsyncIterator

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.domain.chat import ChatRequest, ChatResponse, ChatStreamEvent
from llm_sdk.domain.embeddings import EmbeddingRequest, EmbeddingResponse
from llm_sdk.domain.models import Usage
from llm_sdk.providers.async_base import AsyncBaseLLMClient


class AsyncNoopLLMClient(AsyncBaseLLMClient):
    """
    Async no-op provider.

    Useful for:
    - dev
    - CI tests
    - deterministic behavior
    """

    @property
    def provider_name(self) -> str:
        return "noop"


    async def chat(self, request: ChatRequest) -> ChatResponse:
        content = f"[noop-async] model={request.model} messages={len(request.messages)}"
        return ChatResponse(model=request.model, content=content, usage=Usage(0, 0, 0), raw=None)


    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        vectors: list[list[float]] = []
        for text in request.input:
            vec = [float((ord(c) % 10)) / 10.0 for c in text[:16]]
            vectors.append(vec)
        return EmbeddingResponse(model=request.model, vectors=vectors, raw=None)


    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        msg = (await self.chat(request)).content
        for ch in msg.split(" "):
            yield ChatStreamEvent(delta=ch + " ", done=False)
        yield ChatStreamEvent(delta="", done=True)