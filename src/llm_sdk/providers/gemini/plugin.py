# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.providers.async_base import AsyncBaseLLMClient
from llm_sdk.providers.gemini.async_client import AsyncGeminiLLMClient
from llm_sdk.providers.async_registry import ProviderSpec
from llm_sdk.settings import SDKSettings
from llm_sdk.timeouts import TimeoutConfig


class GeminiProviderFactory():
    """
    Factory for Gemini async provider.

    This is the object exposed through Python entry points, so the SDK can
    discover and load the provider dynamically without core modifications.
    """

    def spec(self) -> ProviderSpec:
        return ProviderSpec(
            name="gemini",
            supports_chat=True,
            supports_embeddings=True,
            supports_streaming=True,
            is_async=True,
        )


    def create(self, settings: SDKSettings) -> AsyncBaseLLMClient:
        """
        Create an AsyncGeminiLLMClient using SDK settings.

        Args:
            settings: SDKSettings resolved by Pydantic Settings.

        Returns:
            AsyncBaseLLMClient instance.
        """
        timeouts = TimeoutConfig()

        return AsyncGeminiLLMClient(
            location=settings.gemini.location,
            scope=settings.gemini.scopes,
            timeouts=timeouts,
        )
