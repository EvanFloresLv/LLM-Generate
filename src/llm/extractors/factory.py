# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from typing import Any

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from src.extractors.base import TokenUsageExtractor
from src.extractors.gemini import GeminiUsageExtractor
from src.extractors.fallback import FallbackUsageExtractor


class TokenUsageExtractorFactory:
    """
    Factory Method:
    Chooses the correct extractor based on provider or response shape.
    """

    @staticmethod
    def create(
        provider: str,
        response: Any
    ) -> TokenUsageExtractor:
        """
        Create a TokenUsageExtractor based on the provider and response.

        Args:
            provider: The name of the LLM provider (e.g., "gemini").
            response: The response object from the LLM.

        Returns:
            A TokenUsageExtractor instance.
        """

        provider = (provider or "").lower()

        if provider == "gemini":
            return GeminiUsageExtractor()

        # Heuristic mode: detect from response
        if getattr(response, "usage_metadata", None) is not None:
            return GeminiUsageExtractor()

        return FallbackUsageExtractor()
