# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
import functools
from typing import Optional

# ---------------------------------------------------------------------
# Internal Application Imports
# ---------------------------------------------------------------------
from src.core.tokens import TokenTracker
from src.extractors.factory import TokenUsageExtractorFactory
from src.schemas.token_schema import TokenUsage


def token_usage(
    provider: str = "unknown",
    model: Optional[str] = None
):
    """
    Generic token usage decorator (SOLID).
    Uses Factory Method to extract TokenUsage from response.

    Args:
        provider: The name of the LLM provider (e.g., "gemini").
        model: The name of the model being used (optional).

    Returns:
        None
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            if isinstance(result, tuple) and len(result) == 2:
                response, input_tokens = result
            else:
                response, input_tokens = result, None

            if response is None:
                raise ValueError(f"{func.__name__} returned None response")

            model_name = model or getattr(self, "_model_name", "unknown")

            extractor = TokenUsageExtractorFactory.create(provider=provider, response=response)
            usage: TokenUsage = extractor.extract(response=response, model=model_name)

            # Optional: add input_tokens if separately computed
            if input_tokens is not None:
                if isinstance(input_tokens, int):
                    usage.tokens.input += int(input_tokens)
                else:
                    usage.tokens.input += int(getattr(input_tokens, "total_tokens", 0) or 0)

            # Recompute total
            usage.total_tokens = (
                usage.tokens.input
                + usage.tokens.output
                + usage.tokens.image
                + usage.tokens.text
                + usage.tokens.thought
            )

            # Persist
            tracker = TokenTracker()
            tracker.increment_call()

            if usage.tokens.input:
                tracker.add_usage(usage.tokens.input, "input")
            if usage.tokens.output:
                tracker.add_usage(usage.tokens.output, "output")
            if usage.tokens.image:
                tracker.add_usage(usage.tokens.image, "image")
            if usage.tokens.text:
                tracker.add_usage(usage.tokens.text, "text")
            if usage.tokens.thought:
                tracker.add_usage(usage.tokens.thought, "thought")

            tracker.save()
            return response, usage

        return wrapper

    return decorator
