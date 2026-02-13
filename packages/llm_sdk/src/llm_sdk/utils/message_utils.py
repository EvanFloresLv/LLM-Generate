# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from typing import Any

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------

from llm_sdk.domain.chat import ChatMessage, Usage


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


def _normalized_messages(messages: list[ChatMessage | tuple[str, str]]) -> list[ChatMessage]:
    """
    Normalize messages to a consistent format.

    Args:
        messages: List of messages to normalize.

    Returns:
        List of normalized ChatMessage objects.
    """
    normalized_messages: list[ChatMessage] = []

    for message in messages:
        if isinstance(message, ChatMessage):
            normalized_messages.append(message)
        else:
            role, content = message
            normalized_messages.append(_msg(role, content))
    return normalized_messages


def extract_token_usage(resp: Any) -> Usage | None:
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
    thought = getattr(usage, "thoughts_token_count", None)

    return Usage(
        prompt_tokens=int(prompt) if prompt is not None else None,
        completion_tokens=int(completion) if completion is not None else None,
        total_tokens=int(total) if total is not None else None,
        thought_tokens=int(thought) if thought is not None else None,
    )