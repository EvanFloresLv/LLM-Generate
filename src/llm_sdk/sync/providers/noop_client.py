
# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Iterator

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.domain.chat import ChatRequest, ChatResponse, ChatStreamEvent
from llm_sdk.domain.embeddings import EmbeddingRequest, EmbeddingResponse
from llm_sdk.domain.models import Usage
from llm_sdk.sync.providers.base import BaseLLMClient


class NoopLLMClient(BaseLLMClient):
    """
    No-op provider for local development and tests.

    Behavior:
    - chat: returns deterministic output
    - embed: returns fake vectors
    - stream_chat: streams deterministic chunks
    """

    @property
    def provider_name(self) -> str:
        return "noop"


    def chat(self, request: ChatRequest) -> ChatResponse:
        total_parts = sum(len(m.normalized_parts()) for m in request.messages)
        content = f"[noop] model={request.model} messages={len(request.messages)} parts={total_parts}"
        return ChatResponse(model=request.model, content=content, usage=Usage(0, 0, 0), raw=None)


    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        vectors: list[list[float]] = []
        for text in request.input:
            # deterministic pseudo-vector
            vec = [float((ord(c) % 10)) / 10.0 for c in text[:16]]
            vectors.append(vec)
        return EmbeddingResponse(model=request.model, vectors=vectors, raw=None)


    def stream_chat(self, request: ChatRequest) -> Iterator[ChatStreamEvent]:
        msg = self.chat(request).content
        for ch in msg.split(" "):
            yield ChatStreamEvent(delta=ch + " ")
        yield ChatStreamEvent(delta="", done=True)
