# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------
from logger.logger import Logger

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.context import Context

from llm_sdk.domain.chat import OutputMimeType, ChatRequest, ChatResponse, ChatStream
from llm_sdk.domain.embeddings import EmbeddingRequest, EmbeddingResponse

from llm_sdk.providers.sync_base import BaseLLMClient as LLMClient
from llm_sdk.providers.sync_registry import ProviderRegistry

from llm_sdk.exceptions import ValidationError
from llm_sdk.retries import with_retries
from llm_sdk.settings import SDKSettings, load_settings

from llm_sdk.utils.message_utils import _normalized_messages


@dataclass(slots=True)
class LLM:
    """
    Main SDK entry point.

    Responsibilities:
    - Provider/model selection via registry
    - Request validation
    - Retries and error normalization
    - Structured logging
    """

    registry: ProviderRegistry
    settings: SDKSettings
    logger: Logger | None = None

    # Internal cache: reuse provider clients (important for HTTP sessions).
    _clients: dict[str, LLMClient] = field(default_factory=dict, init=False, repr=False)


    @classmethod
    def default(
        cls,
        logger: Logger | None = None,
    ) -> "LLM":
        """
        Build a default SDK instance.

        Args:
            log_config: Optional logging configuration.

        Returns:
            LLM
        """
        settings = load_settings()

        return cls(
            registry=ProviderRegistry(),
            settings=settings,
            logger=logger,
        )


    @classmethod
    def from_settings(
        cls,
        settings: SDKSettings,
        *,
        registry: ProviderRegistry | None = None,
        logger: Logger | None = None,
    ) -> "LLM":
        """
        Build an SDK instance from settings.

        Args:
            settings: SDKSettings
            registry: Optional ProviderRegistry

        Returns:
            LLM
        """
        return cls(
            registry=registry or ProviderRegistry(),
            settings=settings,
            logger=logger,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def chat(
        self,
        *,
        messages: list[tuple[str, str]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
        output_schema: dict[str, Any] | None = None,
        output_mime_type: OutputMimeType = "application/json",
    ) -> ChatResponse:
        """
        High-level chat API.

        Args:
            messages: List of (role, content).
            provider: Provider override.
            model: Model override.
            temperature: Sampling temperature.
            max_output_tokens: Output token limit.

        Returns:
            ChatResponse
        """
        prov, mod = self._resolve_provider_and_model(provider, model)
        self.registry.resolve_model(prov, mod)

        req = ChatRequest(
            model=mod,
            messages=_normalized_messages(messages),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            output_mime_type=output_mime_type,
            output_schema=output_schema,
        )

        validate_chat_request(req)

        ctx = Context(provider=prov, model=mod)

        if self.logger is not None:
            self.logger.bind("sync_chat").info(f"chat.request | {asdict(ctx)}")

        client = self._get_client(prov)

        def call() -> ChatResponse:
            return client.chat(req)

        resp = with_retries(fn=call, provider=prov, retry_policy=self.settings.retries)

        if self.logger is not None:
            self.logger.bind("sync_chat").info(f"chat.response | {asdict(ctx)}")

        return resp

    def embed(
        self,
        *,
        input: list[str],
        provider: str | None = None,
        model: str | None = None,
    ) -> EmbeddingResponse:
        """
        High-level embeddings API.

        Args:
            input: List of texts.
            provider: Provider override.
            model: Model override.

        Returns:
            EmbeddingResponse
        """
        prov, mod = self._resolve_provider_and_model(provider, model)
        self.registry.resolve_model(prov, mod)

        req = EmbeddingRequest(model=mod, input=input)
        validate_embedding_request(req)

        ctx = Context(provider=prov, model=mod)
        if self.logger is not None:
            self.logger.bind("sync_embed").info(f"embed.request | {asdict(ctx)}")

        client = self._get_client(prov)

        def call() -> EmbeddingResponse:
            return client.embed(req)

        resp = with_retries(fn=call, provider=prov, retry_policy=self.settings.retries)

        if self.logger is not None:
            self.logger.bind("sync_embed").info(f"embed.response | {asdict(ctx)}")

        return resp


    def stream_chat(
        self,
        *,
        messages: list[tuple[str, str]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
        output_schema: dict[str, Any] | None = None,
        output_mime_type: OutputMimeType = "application/json",
    ) -> ChatStream:
        """
        High-level streaming chat API.

        Notes:
            Streaming retries are not done per-chunk.
            If you want, you can add a "handshake retry" wrapper.

        Args:
            messages: List of (role, content).
            provider: Provider override.
            model: Model override.
            temperature: Sampling temperature.

        Returns:
            ChatStream
        """
        prov, mod = self._resolve_provider_and_model(provider, model)
        self.registry.resolve_model(prov, mod)

        req = ChatRequest(
            model=mod,
            messages=_normalized_messages(messages),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            output_mime_type=output_mime_type,
            output_schema=output_schema,
        )

        validate_chat_request(req)

        client = self._get_client(prov)
        ctx = Context(provider=prov, model=mod)

        if self.logger is not None:
            self.logger.bind("sync_stream_chat").info(f"stream.request | {asdict(ctx)}")

        # No per-chunk retries (safe).
        for event in client.stream_chat(req):
            yield event
            if event.done:
                if self.logger is not None:
                    self.logger.bind("sync_stream_chat").info(f"stream.done | {asdict(ctx)}")
                return

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _resolve_provider_and_model(
        self,
        provider: str | None,
        model: str | None,
    ) -> tuple[str, str]:
        """
        Resolve provider/model using defaults from settings.

        Args:
            provider: Optional provider override
            model: Optional model override

        Returns:
            (provider, model)
        """
        prov = provider or self.settings.default_provider
        mod = model or self.settings.default_model

        if not prov:
            raise ValidationError("provider cannot be empty")
        if not mod:
            raise ValidationError("model cannot be empty")

        return prov, mod


    def _get_client(self, provider: str) -> LLMClient:
        """
        Get or create a cached client for a provider.

        Args:
            provider: Provider name

        Returns:
            LLMClient
        """
        cached = self._clients.get(provider)
        if cached is not None:
            return cached

        spec = self.registry.get(provider)
        client = spec.factory()

        self._clients[provider] = client
        return client


# ---------------------------------------------------------------------
# Validators (should live in llm_sdk/validators/*.py)
# ---------------------------------------------------------------------

def validate_chat_request(request: ChatRequest) -> None:
    """
    Validate chat request.

    Args:
        request: ChatRequest

    Raises:
        ValidationError
    """
    if not request.messages:
        raise ValidationError("messages cannot be empty")

    for m in request.messages:
        parts = m.normalized_parts()

        if not parts:
            raise ValidationError("message parts cannot be empty")

        has_any = False

        for part in parts:

            if part.type == "text" and part.text and part.text.strip():
                has_any = True
            elif part.type in ("image_url", "file_uri") and (part.url or part.uri):
                has_any = True
            elif part.type == "image_bytes" and part.data:
                has_any = True

        if not has_any:
            raise ValidationError("message parts must contain at least one valid part")

    if request.temperature < 0.0 or request.temperature > 2.0:
        raise ValidationError("temperature must be between 0 and 2")


def validate_embedding_request(request: EmbeddingRequest) -> None:
    """
    Validate embedding request.

    Args:
        request: EmbeddingRequest

    Raises:
        ValidationError
    """
    if not request.input:
        raise ValidationError("input cannot be empty")

    if any(not x.strip() for x in request.input):
        raise ValidationError("input texts cannot be empty")
