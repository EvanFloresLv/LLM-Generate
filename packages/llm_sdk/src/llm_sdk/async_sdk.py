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
from llm_sdk.lifecycle import AsyncResourceManager

from llm_sdk.domain.chat import ChatMessage, ChatRequest, ChatResponse, ChatStreamEvent
from llm_sdk.domain.embeddings import EmbeddingRequest, EmbeddingResponse

from llm_sdk.providers.async_base import AsyncBaseLLMClient
from llm_sdk.providers.async_registry import ProviderRegistry

from llm_sdk.plugin_loader import load_provider_plugins
from llm_sdk.context import Context
from llm_sdk.exceptions import ValidationError
from llm_sdk.settings import SDKSettings, load_settings
from llm_sdk.retries import RetryPolicy, with_async_retries
from llm_sdk.utils.message_utils import _normalized_messages


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

    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    logger: Logger | None = None


    @classmethod
    def default(
        cls,
        load_plugins: bool = False,
        logger: Logger | None = None,
    ) -> "AsyncLLM":
        """
        Build default AsyncLLM instance.

        Args:
            load_plugins: Whether to load provider plugins.
            logger: Optional logger instance. If None, no logs are emitted.

        Returns:
            AsyncLLM
        """
        registry = ProviderRegistry()
        settings = load_settings()

        if load_plugins:
            registry.load_plugins()

        return cls(
            registry=registry,
            settings=settings,
            logger=logger
        )


    def load_plugins(self) -> None:
        """
        Load provider plugins using entrypoints.
        """
        result = load_provider_plugins(self.registry, self.settings)

        if self.logger is not None:
            log = self.logger.bind("load_plugins")
            log.info(
                f"plugins.loaded | loaded: {result.loaded}, failed: {result.failed}"
            )


    async def aclose(self) -> None:
        """
        Close all cached provider resources.
        """
        await self.resources.aclose()


    async def chat(
        self,
        *,
        messages: list[tuple[str, str]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> ChatResponse:
        """
        Async chat API.

        Args:
            messages: The chat messages to send.
            provider: The provider name.
            model: The model name.
            temperature: The temperature to use for the chat.
            max_output_tokens: The maximum number of output tokens.

        Returns:
            ChatResponse: The chat response.
        """
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
            messages=_normalized_messages(messages),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        self._validate_chat(req)

        client = await self._get_provider_client(prov)
        ctx = Context(provider=prov, model=mod)

        if self.logger is not None:
            self.logger.bind("async_chat").info(f"chat.request | {asdict(ctx)}")

        async def _call() -> ChatResponse:
            return await client.chat(req)

        resp = await with_async_retries(
            fn=_call, provider=prov, retry_policy=self.retry_policy
        )

        if self.logger is not None:
            self.logger.bind("async_chat").info(f"chat.response | {asdict(ctx)}")

        return resp


    async def embed(
        self,
        *,
        input: list[str],
        provider: str | None = None,
        model: str | None = None,
    ) -> EmbeddingResponse:
        """
        Async embedding API.

        Args:
            input: The input text to embed.
            provider: The provider name.
            model: The model name.

        Returns:
            EmbeddingResponse: The embedding response.
        """
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

        if self.logger is not None:
            self.logger.bind("async_embed").info(f"embed.request | {asdict(ctx)}")

        async def _call() -> EmbeddingResponse:
            return await client.embed(req)

        resp = await with_async_retries(
            fn=_call, provider=prov, retry_policy=self.retry_policy
        )

        if self.logger is not None:
            self.logger.bind("async_embed").info(f"embed.response | {asdict(ctx)}")

        return resp


    async def stream_chat(
        self,
        *,
        messages: list[tuple[str, str]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[ChatStreamEvent]:
        """
        Async streaming chat API.

        Args:
            message: The chat messages to send.
            provider: The provider name.
            model: The model name.
            temperature: The temperature to use for the chat.

        Returns:
            AsyncIterator[ChatStreamEvent]: The streaming chat events.
        """
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
            messages=_normalized_messages(messages),
            temperature=temperature,
        )

        client = await self._get_provider_client(prov)
        ctx = Context(provider=prov, model=mod)

        if self.logger is not None:
            self.logger.bind("async_stream_chat").info(
                f"stream.request | {asdict(ctx)}"
            )

        async for event in client.stream_chat(req):
            yield event
            if event.done:
                if self.logger is not None:
                    self.logger.bind("async_stream_chat").info(
                        f"stream.done | {asdict(ctx)}"
                    )
                return


    async def _get_provider_client(self, provider: str) -> AsyncBaseLLMClient:
        """
        Get cached provider client or create a new one.

        Args:
            provider: The provider name.

        Returns:
            AsyncBaseLLMClient: The provider client.
        """
        cached = self.resources.get_cached(provider)
        if cached is not None:
            return cached

        factory = self.registry.get(provider)
        client = factory.create(self.settings)

        if not isinstance(client, AsyncBaseLLMClient):
            raise TypeError(
                f"Provider factory '{provider}' returned invalid client type: {type(client)}"
            )

        self.resources.set_cached(provider, client)
        return client


    def _validate_chat(self, request: ChatRequest) -> None:
        """
        Validate chat request.

        Args:
            request: ChatRequest

        Returns:
            None
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

        Returns:
            None
        """
        if not request.input:
            raise ValidationError("input cannot be empty")

        if any(not x.strip() for x in request.input):
            raise ValidationError("input texts cannot be empty")
