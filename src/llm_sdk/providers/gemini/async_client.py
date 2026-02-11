# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any, AsyncIterator

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------
from google import genai
from google.auth import default
from google.genai.types import Content, Part

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.domain.chat import ChatMessage, ChatRequest, ChatResponse, ChatStreamEvent
from llm_sdk.domain.embeddings import EmbeddingRequest, EmbeddingResponse
from llm_sdk.domain.models import Usage
from llm_sdk.exceptions import ProviderError
from llm_sdk.providers.async_base import AsyncBaseLLMClient
from llm_sdk.timeouts import TimeoutConfig


class AsyncGeminiLLMClient(AsyncBaseLLMClient):
    """
    Async Gemini provider implementation using google-genai (Vertex AI).

    Supports:
    - chat completions
    - embeddings
    - streaming chat (async)
    """

    def __init__(
        self,
        *,
        location: str,
        scope: list[str] | None = None,
        timeouts: TimeoutConfig,
    ) -> None:
        self._credentials, self._project = default(scopes=scope)
        self._location = location
        self._timeouts = timeouts

        self._client = genai.Client(
            vertexai=True,
            project=self._project,
            location=self._location,
            credentials=self._credentials,
        )

        # Async client handle
        self._aio = self._client.aio


    @property
    def provider_name(self) -> str:
        return "gemini"


    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Execute a chat completion request.

        Args:
            request: ChatRequest with model, messages and optional params.

        Returns:
            ChatResponse with normalized content, usage and raw provider payload.
        """
        contents = self._to_gemini_contents(request.messages)

        try:
            resp = await self._aio.models.generate_content(
                model=request.model,
                contents=contents,
                config={
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_output_tokens,
                },
            )
        except Exception as e:
            raise ProviderError("gemini", f"provider error: {e}", is_retryable=True) from e

        text = getattr(resp, "text", None)
        if not text:
            text = self._extract_text_fallback(resp)

        usage = self._extract_usage(resp)
        raw = self._safe_raw(resp)

        return ChatResponse(model=request.model, content=text or "", usage=usage, raw=raw)


    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        """
        Stream chat completion deltas as they arrive.

        Args:
            request: ChatRequest.

        Yields:
            ChatStreamEvent deltas until done=True.
        """
        contents = self._to_gemini_contents(request.messages)

        try:
            stream = await self._aio.models.generate_content_stream(
                model=request.model,
                contents=contents,
                config={
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_output_tokens,
                },
            )
        except Exception as e:
            raise ProviderError("gemini", f"provider error: {e}", is_retryable=True) from e

        try:
            async for chunk in stream:
                delta = getattr(chunk, "text", None)
                if delta:
                    yield ChatStreamEvent(delta=delta, done=False)

            yield ChatStreamEvent(delta="", done=True)

        except Exception as e:
            raise ProviderError("gemini", f"stream error: {e}", is_retryable=True) from e


    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings.

        Args:
            request: EmbeddingRequest with model and input texts.

        Returns:
            EmbeddingResponse with vectors aligned with input order.
        """
        try:
            resp = await self._aio.models.embed_content(
                model=request.model,
                contents=request.input,
            )
        except Exception as e:
            raise ProviderError("gemini", f"provider error: {e}", is_retryable=True) from e

        vectors = self._extract_embeddings(resp)
        raw = self._safe_raw(resp)

        return EmbeddingResponse(model=request.model, vectors=vectors, raw=raw)

    # -------------------------
    # Helpers
    # -------------------------

    def _to_gemini_contents(self, messages: list[ChatMessage]) -> list[Content]:
        """
        Convert SDK messages into Gemini Contents.

        Supports:
        - text
        - image_url (as URI)
        - file_uri
        - image_bytes

        Args:
            messages: List of ChatMessage from SDK domain.

        Returns:
            List of Gemini Content objects.
        """
        contents: list[Content] = []

        for msg in messages:
            role = msg.role
            if role == "assistant":
                role = "model"

            parts_out: list[Part] = []

            for part in msg.normalized_parts():
                if part.type == "text":
                    parts_out.append(Part.from_text(text=part.text or ""))

                elif part.type == "image_url":
                    parts_out.append(Part.from_uri(file_uri=part.url or ""))

                elif part.type == "file_uri":
                    parts_out.append(Part.from_uri(file_uri=part.uri or ""))

                elif part.type == "image_bytes":
                    if not part.mime_type:
                        raise ProviderError(
                            "gemini",
                            "image_bytes requires mime_type (e.g. image/png)",
                            is_retryable=False,
                        )
                    if not part.data:
                        raise ProviderError("gemini", "image_bytes requires data", is_retryable=False)

                    parts_out.append(
                        Part.from_bytes(
                            data=part.data,
                            mime_type=part.mime_type,
                        )
                    )

                else:
                    raise ProviderError("gemini", f"unsupported part type: {part.type}", is_retryable=False)

            contents.append(Content(role=role, parts=parts_out))

        return contents


    def _extract_embeddings(self, resp: Any) -> list[list[float]]:
        """
        Extract embeddings from google-genai response.

        Args:
            resp: Provider response.

        Returns:
            list[list[float]] aligned with input order.
        """
        embeddings = getattr(resp, "embeddings", None)
        if embeddings:
            out: list[list[float]] = []
            for e in embeddings:
                values = getattr(e, "values", None)
                if values is None:
                    values = getattr(e, "embedding", None)
                out.append(list(values or []))
            return out

        values = getattr(resp, "values", None)
        if values:
            return [list(values)]

        return []


    def _extract_usage(self, resp: Any) -> Usage | None:
        """
        Extract token usage if available.

        Args:
            resp: Provider response.

        Returns:
            Usage if present, else None.
        """
        usage = getattr(resp, "usage_metadata", None)
        if not usage:
            return None

        prompt = getattr(usage, "prompt_token_count", None)
        completion = getattr(usage, "candidates_token_count", None)
        total = getattr(usage, "total_token_count", None)

        return Usage(
            prompt_tokens=int(prompt) if prompt is not None else None,
            completion_tokens=int(completion) if completion is not None else None,
            total_tokens=int(total) if total is not None else None,
        )


    def _extract_text_fallback(self, resp: Any) -> str:
        """
        Fallback extraction if resp.text is missing.

        Args:
            resp: Provider response.

        Returns:
            Extracted text or empty string.
        """
        candidates = getattr(resp, "candidates", None)
        if not candidates:
            return ""

        first = candidates[0]
        content = getattr(first, "content", None)
        if not content:
            return ""

        parts = getattr(content, "parts", None) or []
        out: list[str] = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                out.append(t)
        return "".join(out)


    def _safe_raw(self, obj: Any) -> dict[str, Any]:
        """
        Convert response to a safe JSON-like dict for debugging.

        Args:
            obj: Provider response object.

        Returns:
            JSON-like dict.
        """
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            try:
                raw = dump()
                if isinstance(raw, dict):
                    return raw
            except Exception:
                return {}

        return {"repr": repr(obj)}
