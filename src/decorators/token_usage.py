# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
import types
import functools
from typing import Optional

# ---------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------
from google.genai import types

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from src.core.tokens import TokenTracker


def gemini_token_usage(
    model: Optional[str] = None
):
    """
    Decorator to record Gemini token usage.
    Requires TokenTracker to be initialized earlier with config.

    Args:
        model: Optional model name to track usage for a specific model.

    Return:
        Decorated function that tracks token usage.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            response = func(*args, **kwargs)

            usage_metadata = getattr(response, "usage_metadata", None)
            if not usage_metadata:
                return response

            tracker = TokenTracker()  # already initialized elsewhere
            tracker.increment_call()

            # Output tokens
            candidates = getattr(usage_metadata, "candidates_token_count", 0) or 0
            if candidates:
                tracker.add_usage(candidates, "output_tokens")

            # Prompt modality tokens
            prompt_details = getattr(usage_metadata, "prompt_tokens_details", []) or []
            for part in prompt_details:
                modality = getattr(part, "modality", None)
                token_count = getattr(part, "token_count", 0) or 0

                if modality == types.MediaModality.IMAGE:
                    tracker.add_usage(token_count, "image_tokens")
                elif modality == types.MediaModality.TEXT:
                    tracker.add_usage(token_count, "text_tokens")

            # Thought tokens
            thoughts = getattr(usage_metadata, "thoughts_token_count", 0) or 0
            if thoughts:
                tracker.add_usage(thoughts, "thought_tokens")

            tracker.save()
            return response

        return wrapper

    return decorator
