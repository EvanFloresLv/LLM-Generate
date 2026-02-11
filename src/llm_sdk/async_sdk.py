# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import AsyncIterator

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------
from logger.logger import Logger

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.async_retries import AsyncRetryPolicy, with_async_retries
from llm_sdk.context import Context
from llm_sdk.domain.chat import ChatMessage, ChatRequest, ChatResponse, ChatStreamEvent
from llm_sdk.domain.embeddings import EmbeddingRequest, EmbeddingResponse
from llm_sdk.exceptions import ValidationError
from llm_sdk.lifecycle import AsyncResourceManager

from llm_sdk.providers.async_base import AsyncBaseLLMClient
from llm_sdk.providers.async_registry import ProviderRegistry

from llm_sdk.settings import SDKSettings, load_settings
from llm_sdk.plugin_loader import load_provider_plugins

Logger().configure()


@dataclass(frozen=True)
class AsyncLLM:
    """
    Async SDK entry point.

    Features:
    - provider/model resolution via registry
    - retries + logging + validation in core
    - provider plugins via entrypoints
    - provider client caching (connection reuse)
    """

    registry: ProviderRegistry
    settings: SDKSettings
    resources: AsyncResourceManager = field(default_factory=AsyncResourceManager)

    retry_policy: AsyncRetryPolicy = field(default_factory=AsyncRetryPolicy)

    logger: Logger = Logger()

    @classmethod
    def default(cls) -> "AsyncLLM":
        """
        Build default AsyncLLM instance.

        Returns:
            AsyncLLM
        """
        return cls(registry=ProviderRegistry(), settings=load_settings())


    def load_plugins(self) -> None:
        """
        Load provider plugins using entrypoints.

        Returns:
            None
        """
        result = load_provider_plugins(self.registry, self.settings)

        plugin_logger = self.logger.bind("load_plugins")
        plugin_logger.info(f"plugins.loaded | loaded: {result.loaded}, failed: {result.failed}")


    async def aclose(self) -> None:
        """
        Close all cached provider resources.

        Returns:
            None
        """
        await self.resources.aclose()


    async def chat(
        self,
        *,
        messages: list[tuple[str, str]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None
    ):
        """
        Async chat API.

        Args:
            messages: List of (role, content).
            provider: Provider override.
            model: Model override.
            temperature: Sampling temperature.
            max_output_tokens: Optional token limit.

        Returns:
            ChatResponse
        """
        chat_log = self.logger.bind("async_chat")

        prov = provider or self.settings.default_provider
        mod = model or self.settings.default_model

        factory = self.registry.get(prov)
        spec = factory.spec()

        if not spec.supports_chat:
            raise ValidationError(f"provider '{prov}' does not support chat")

        if not spec.is_async:
            raise ValidationError(f"provider '{prov}' is not async (is_async=False)")

        req = ChatRequest(
            model=mod,
            messages=[_msg(role, content) for role, content in messages],
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )

        self._validate_chat(req)

        client = await self._get_provider_client(prov)
        ctx = Context(provider=prov, model=mod)

        chat_log.info(f"chat.request | {asdict(ctx)}")

        async def chat() -> ChatResponse:
            return await client.chat(req)

        resp = await with_async_retries(fn=chat, provider=prov, retry_policy=self.retry_policy)
        chat_log.info(f"chat.response | {asdict(ctx)}")

        return resp


    async def embed(
        self,
        *,
        input: list[str],
        provider: str | None = None,
        model: str | None = None
    ) -> EmbeddingResponse:
        """
        Async embedding API.

        Args:
            input: List of input texts.
            provider: Provider override.
            model: Model override.

        Returns:
            EmbeddingResponse
        """
        embed_log = self.logger.bind("async_embed")

        prov = provider or self.settings.default_provider
        mod = model or self.settings.default_model

        factory = self.registry.get(prov)
        spec = factory.spec()

        if not spec.supports_embeddings:
            raise ValidationError(f"provider '{prov}' does not support embeddings")

        if not spec.is_async:
            raise ValidationError(f"provider '{prov}' is not async (is_async=False)")

        req = EmbeddingRequest(model=mod, input=input)
        self._validate_embed(req)

        client = await self._get_provider_client(prov)
        ctx = Context(provider=prov, model=mod)

        embed_log.info(f"embed.request | {asdict(ctx)}")

        async def embed() -> EmbeddingResponse:
            return await client.embed(req)

        resp = await with_async_retries(fn=embed, provider=prov, retry_policy=self.retry_policy)
        embed_log.info(f"embed.response | {asdict(ctx)}")

        return resp


    async def stream_chat(
        self,
        *,
        message: list[tuple[str, str]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[ChatStreamEvent]:
        """
        Async streaming chat API.

        Args:
            messages: List of (role, content).
            provider: Provider override.
            model: Model override.
            temperature: Sampling temperature.

        Yields:
            ChatStreamEvent
        """
        stream_log = self.logger.bind("async_stream_chat")

        prov = provider or self.settings.default_provider
        mod = model or self.settings.default_model

        factory = self.registry.get(prov)
        spec = factory.spec()

        if not spec.supports_streaming:
            raise ValidationError(f"provider '{prov}' does not support streaming")

        if not spec.is_async:
            raise ValidationError(f"provider '{prov}' is not async (is_async=False)")

        req = ChatRequest(
            model=mod,
            messages=[_msg(role, content) for role, content in message],
            temperature=temperature
        )

        client = await self._get_provider_client(prov)

        ctx = Context(provider=prov, model=mod)
        stream_log.info(f"stream.request | {asdict(ctx)}")

        async for event in client.stream_chat(req):
            yield event
            if event.done:
                stream_log.info(f"stream.done | {asdict(ctx)}")
                return


    async def _get_provider_client(self, provider: str) -> AsyncBaseLLMClient:
        """
        Get cached provider client or create a new one.

        Args:
            provider: Provider name.

        Returns:
            AsyncBaseLLMClient
        """
        cached = self.resources.get_cached(provider)
        if cached is not None:
            return cached

        factory = self.registry.get(provider)
        client = factory.create(self.settings)

        # (Opcional) Validación extra de contrato
        if not isinstance(client, AsyncBaseLLMClient):
            raise TypeError(
                f"Provider factory '{provider}' returned invalid client type: {type(client)}"
            )

        self.resources.get_cached(provider)
        return client


    def _validate_chat(self, request: ChatRequest) -> None:
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


    def _validate_embed(self, request: EmbeddingRequest) -> None:
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


def _msg(role: str, content: str) -> ChatMessage:
    """
    Create a chat message.

    Args:
        role: The role of the message sender (user/system).
        content: The content of the message.

    Returns:
        ChatMessage
    """
    return ChatMessage(role=role, content=content)