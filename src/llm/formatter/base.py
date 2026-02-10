# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
import imghdr
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


class BasePromptFormatter(ABC):
    """
    Strategy interface for provider-specific prompt formatting.
    """

    # Map of image file extensions to MIME types
    IMAGE_MIME_MAP = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }

    @abstractmethod
    def format(
        self,
        prompt: Dict[str, str],
        image_urls: Optional[List[str]] = None,
    ) -> Any:
        """
        Formats the prompt and associated image URLs for the specific provider.

        Args:
            prompt (Dict[str, str]): The prompt to format.
            image_urls (Optional[List[str]]): The image URLs to include.

        Returns:
            Any: The formatted prompt.
        """
        raise NotImplementedError

    # ---------- Shared helpers ----------

    def _validate_prompt(self, prompt: Dict[str, str]) -> None:
        """
        Validates the prompt structure (system and user).

        Args:
            prompt (Dict[str, str]): The prompt to validate.

        Returns:
            None
        """

        if not isinstance(prompt, dict):
            raise ValueError("Prompt must be a dictionary")

        if not prompt.get("system") and not prompt.get("user"):
            raise ValueError("Prompt must contain 'system' or 'user'")


    def _get_mime_type(self, url: str) -> str:
        """
        Returns the MIME type for a given image URL.

        Args:
            url (str): The image URL.

        Returns:
            str: The MIME type of the image.
        """

        if "." not in url:
            return "image/jpeg"
        return self.IMAGE_MIME_MAP.get(url.split(".")[-1].lower(), "image/jpeg")


    def _get_mime_type_from_bytes(self, image_bytes: bytes) -> str:
        """
        Returns the MIME type for a given image byte array.

        Args:
            image_bytes (bytes): The image byte array.

        Returns:
            str: The MIME type of the image.
        """

        img_type = imghdr.what(None, image_bytes)
        return f"image/{img_type or 'jpeg'}"