# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.domain.models import Usage


ChatPartType = Literal[
    "text",
    "image_url",
    "image_bytes",
    "file_uri",
]


@dataclass(frozen=True, slots=True)
class ChatPart:
    """
    A single multimodal message part.

    Supported types:
    - text
    - image_url
    - image_bytes
    - file_uri

    Args:
        type: Part type.
        text: Text content (type=text).
        url: Image URL (type=image_url).
        data: Raw bytes (type=image_bytes).
        mime_type: Required for image_bytes (e.g. image/png).
        uri: File URI (type=file_uri). Example: gs://bucket/file.png
        metadata: Optional provider-agnostic metadata.
    """

    type: ChatPartType
    text: str | None = None
    url: str | None = None
    data: bytes | None = None
    mime_type: str | None = None
    uri: str | None = None
    metadata: dict[str, Any] | None = None


    @staticmethod
    def from_text(text: str) -> "ChatPart":
        """
        Create a text part.

        Args:
            text: Text.

        Returns:
            ChatPart
        """
        return ChatPart(type="text", text=text)


    @staticmethod
    def from_image_url(url: str) -> "ChatPart":
        """
        Create an image url part.

        Args:
            url: Public image URL.

        Returns:
            ChatPart
        """
        return ChatPart(type="image_url", url=url)


    @staticmethod
    def from_image_bytes(data: bytes, mime_type: str) -> "ChatPart":
        """
        Create an image bytes part.

        Args:
            data: Raw bytes.
            mime_type: e.g. image/png

        Returns:
            ChatPart
        """
        return ChatPart(type="image_bytes", data=data, mime_type=mime_type)


    @staticmethod
    def from_file_uri(uri: str) -> "ChatPart":
        """
        Create a file uri part.

        Args:
            uri: File URI (gs://...).

        Returns:
            ChatPart
        """
        return ChatPart(type="file_uri", uri=uri)



@dataclass(frozen=True, slots=True)
class ChatMessage:
    """
    A normalized chat message.

    Backwards compatible:
    - You can still use role + content.
    - If parts is provided, it takes precedence.

    Args:
        role: system/user/assistant.
        content: Legacy plain text.
        parts: Multimodal parts.
    """

    role: str
    content: str = ""
    parts: list[ChatPart] | None = None


    def is_multimodal(self) -> bool:
        """
        Returns True if message uses parts.

        Returns:
            bool
        """
        return bool(self.parts)


    def normalized_parts(self) -> list[ChatPart]:
        """
        Returns parts if provided, else converts content to a single text part.

        Returns:
            list[ChatPart]
        """
        if self.parts and len(self.parts) > 0:
            return self.parts
        return [ChatPart.from_text(self.content)]


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """
    Chat completion request.

    Args:
        model: Model name.
        messages: Chat messages.
        temperature: Sampling temperature.
        max_output_tokens: Optional limit.
        metadata: Free-form metadata.
    """

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_output_tokens: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """
    Normalized chat response.

    Args:
        model: Model used.
        content: Assistant output.
        usage: Token usage.
        raw: Provider raw payload for debugging.
    """

    model: str
    content: str
    usage: Usage | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ChatStreamEvent:
    """
    Streaming event.

    Args:
        delta: Token chunk (may be empty).
        done: True if stream completed.
    """

    delta: str
    done: bool = False


ChatStream = Iterator[ChatStreamEvent]
