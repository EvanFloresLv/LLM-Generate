# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------
from logger.logger import Logger

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.domain.chat import ChatMessage, ChatRequest, ChatResponse, ChatStream
from llm_sdk.domain.embeddings import EmbeddingRequest, EmbeddingResponse
from llm_sdk.context import Context
from llm_sdk.exceptions import ValidationError
from llm_sdk.registry import ProviderRegistry
from llm_sdk.retries import with_retries
from llm_sdk.settings import SDKSettings, load_settings


Logger().configure()

@dataclass(slots=True)
class LLM:
    """
    Main SDK entry point.

    This class:
    - selects provider + model via registry
    - validates requests
    - applies retries
    - logs structured events
    """

    registry: ProviderRegistry
    settings: SDKSettings
    logger: Logger = Logger()

    @classmethod
    def default(cls) -> "LLM":
        """
        Build a default SDK instance.

        Returns:
            LLM
        """
        return cls(registry=ProviderRegistry(), settings=load_settings())


    def chat(
        self,
        *,
        messages: list[tuple[str, str]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
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
        chat_log = self.logger.bind("chat")

        prov = provider or self.settings.default_provider
        mod = model or self.settings.default_model

        self.registry.resolve_model(prov, mod)

        req = ChatRequest(
            model=mod,
            messages=[_msg(role, content) for role, content in messages],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        self._validate_chat(req)

        client = self.registry.get(prov).factory()
        ctx = Context(provider=prov, model=mod)

        chat_log.info(f"chat.request | {asdict(ctx)}")

        def call() -> ChatResponse:
            return client.chat(req)

        resp = with_retries(fn=call, provider=prov, retry_policy=self.settings.retries)

        chat_log.info(f"chat.response | {asdict(ctx)}")
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
        embed_log = self.logger.bind("embed")

        prov = provider or self.settings.default_provider
        mod = model or self.settings.default_model

        self.registry.resolve_model(prov, mod)

        req = EmbeddingRequest(model=mod, input=input)
        self._validate_embed(req)

        client = self.registry.get(prov).factory()
        ctx = Context(provider=prov, model=mod)

        embed_log.info(f"embed.request | {asdict(ctx)}")

        def call() -> EmbeddingResponse:
            return client.embed(req)

        resp = with_retries(fn=call, provider=prov, retry_policy=self.settings.retries)

        embed_log.info(f"embed.response | {asdict(ctx)}")
        return resp


    def stream_chat(
        self,
        *,
        messages: list[tuple[str, str]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> ChatStream:
        """
        High-level streaming chat API.

        Args:
            messages: List of (role, content).
            provider: Provider override.
            model: Model override.
            temperature: Sampling temperature.

        Returns:
            Iterator[ChatStreamEvent]
        """
        stream_chat_log = self.logger.bind("stream_chat")

        prov = provider or self.settings.default_provider
        mod = model or self.settings.default_model

        self.registry.resolve_model(prov, mod)

        req = ChatRequest(
            model=mod,
            messages=[_msg(role, content) for role, content in messages],
            temperature=temperature,
        )

        self._validate_chat(req)

        client = self.registry.get(prov).factory()
        ctx = Context(provider=prov, model=mod)

        stream_chat_log.info(f"stream.request | {asdict(ctx)}")

        # streaming: no retries at chunk level (simpler + safe)
        for event in client.stream_chat(req):
            yield event
            if event.done:
                stream_chat_log.info(f"stream.done | {asdict(ctx)}")
                return


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


def _msg(role: str, content: str):
    """
    Create a chat message.

    Args:
        role: The role of the message sender (user/system).
        content: The content of the message.

    Returns:
        ChatMessage
    """

    return ChatMessage(role=role, content=content)