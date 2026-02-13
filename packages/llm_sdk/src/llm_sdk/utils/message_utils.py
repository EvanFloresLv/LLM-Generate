# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------

from llm_sdk.domain.chat import ChatMessage


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