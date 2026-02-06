# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from src.schemas.token_schema import TokenUsage


class TokenUsageExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        response: Any,
        model: str
    ) -> TokenUsage:
        """
        Extract token usage information from the response.

        Args:
            response: The response object from the API call.
            model: The model name being used.

        Returns:
            TokenUsage: The extracted token usage information.
        """
        raise NotImplementedError
