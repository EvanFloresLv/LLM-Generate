# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
import abc
from typing import Any, List


class LLMProvider(abc.ABC):
    """Abstract base class for LLM services."""

    @abc.abstractmethod
    def initialize_client(self) -> Any:
        """Initialize and return the LLM client."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_response(self, contents: List[Any], **kwargs) -> str:
        """Generate a response given structured contents."""
        raise NotImplementedError