# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from src.formatter.base import BasePromptFormatter
from src.formatter.gemini import GeminiPromptFormatter

class FormatterFactory:
    """
    Factory for resolving the correct formatter per provider.
    """

    _REGISTRY = {
        "gemini": GeminiPromptFormatter,
    }

    @classmethod
    def get(
        cls,
        provider_name: str
    ) -> BasePromptFormatter:
        """
        Get the appropriate formatter for the specified provider.

        Args:
            provider_name (str): The name of the provider to get the formatter for.

        Returns:
            BasePromptFormatter: The formatter for the specified provider.
        """
        try:
            return cls._REGISTRY[provider_name.lower()]()
        except KeyError:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")


class Formatter:
    """
    Facade for prompt formatting across LLM providers.
    """

    @staticmethod
    def for_provider(
        provider_name: str,
        *,
        prompt: Dict[str, str],
        image_urls: Optional[List[str]] = None,
    ) -> Any:
        """
        Formats the prompt and associated image URLs for the specified provider.

        Args:
            provider_name (str): The name of the provider to format the prompt for.
            prompt (Dict[str, str]): The prompt to format.
            image_urls (Optional[List[str]]): The image URLs to include.

        Returns:
            Any: The formatted prompt for the specified provider.
        """
        formatter = FormatterFactory.get(provider_name)
        return formatter.format(prompt, image_urls)


if __name__ == "__main__":
    try:
        prompt = {
            "system": "System instructions",
            "user": "User instructions",
            "imagenes": [
                "https://ss202.liverpool.com.mx/lg/1132791879.jpg"
            ]
        }

        contents = Formatter.for_provider("gemini", prompt=prompt, image_urls=prompt.get("imagenes", []))

    except Exception as e:
        raise e