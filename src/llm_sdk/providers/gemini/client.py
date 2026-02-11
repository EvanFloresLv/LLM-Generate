# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Iterator

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
from llm_sdk.providers.base import BaseLLMClient
from llm_sdk.timeouts import TimeoutConfig


class GeminiLLMClient(BaseLLMClient):
    """
    Gemini provider implementation using google-genai (Vertex AI).

    Supports:
    - chat completions
    - embeddings
    - streaming chat
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

    @property
    def provider_name(self) -> str:
        return "gemini"


    def chat(self, request: ChatRequest) -> ChatResponse:
        contents = self._to_gemini_contents(request.messages)

        try:
            resp = self._client.models.generate_content(
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
            # Fallback: try extracting from candidates
            text = self._extract_text_fallback(resp)

        usage = self._extract_usage(resp)

        raw = self._safe_raw(resp)
        return ChatResponse(model=request.model, content=text or "", usage=usage, raw=raw)


    def stream_chat(self, request: ChatRequest) -> Iterator[ChatStreamEvent]:
        contents = self._to_gemini_contents(request.messages)

        try:
            stream = self._client.models.generate_content_stream(
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
            for chunk in stream:
                delta = getattr(chunk, "text", None)
                if delta:
                    yield ChatStreamEvent(delta=delta, done=False)
            yield ChatStreamEvent(delta="", done=True)
        except Exception as e:
            raise ProviderError("gemini", f"stream error: {e}", is_retryable=True) from e


    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            resp = self._client.models.embed_content(
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
                    # Gemini supports URI (http/https) via from_uri
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

        Returns:
            list[list[float]] aligned with input order
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

        # Fallback: single embedding
        values = getattr(resp, "values", None)
        if values:
            return [list(values)]

        return []


    def _extract_usage(self, resp: Any) -> Usage | None:
        """
        Extract token usage if available.

        Gemini/Vertex provides usage in some responses.
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
        """
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            try:
                raw = dump()
                if isinstance(raw, dict):
                    return raw
            except Exception:
                return {}

        # fallback: repr
        return {"repr": repr(obj)}


if __name__ == "__main__":

    llm = GeminiLLMClient(location="us-central1", timeouts=TimeoutConfig())

    resp = llm.chat(
        ChatRequest(
            model="gemini-2.0-flash",
            messages=[
                ChatMessage(role="user", content="Write a haiku."),
            ],
            temperature=0.5,
            max_output_tokens=50,
        )
    )

    print(resp.content)