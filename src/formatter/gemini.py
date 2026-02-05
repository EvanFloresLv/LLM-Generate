# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------
from google.genai.types import Content, Part

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from src.storage.gcs import GCS
from src.formatter.base import BasePromptFormatter


class GeminiPromptFormatter(BasePromptFormatter):
    """
    Gemini-specific prompt formatter with deterministic multimodal handling.
    """

    def __init__(
        self,
        config: Any,
        gcs: Optional[GCS] = None
    ) -> None:
        """
        Initialize the GeminiPromptFormatter.

        Args:
            config: Configuration settings for the formatter.
            gcs: Optional GCS client for handling image uploads.

        Returns:
            None
        """

        if config is None:
            raise ValueError("Config is required")

        self.config = config
        self.gcs = gcs

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def format(
        self,
        prompt: Dict[str, str],
        image_urls: Optional[List[str]] = None,
    ) -> List[Content]:
        """
        Build Gemini Content objects from structured prompt and optional images.

        Args:
            prompt: Structured prompt containing user and system messages.
            image_urls: Optional list of image URLs to include in the request.

        Returns:
            List of Gemini Content objects.
        """
        self._validate_prompt(prompt)

        image_urls = image_urls or []
        contents: List[Content] = []

        # ---------------- system ----------------
        system_text = prompt.get("system")
        if system_text:
            contents.append(
                Content(
                    role="system",
                    parts=[Part.from_text(text=str(system_text))],
                )
            )

        # ---------------- user ----------------
        user_text = prompt.get("user")
        if not user_text and not image_urls:
            return contents  # nothing to send

        parts: List[Part] = []

        if user_text:
            parts.append(Part.from_text(text=str(user_text)))

        # Attach images
        for url in image_urls:
            part = self._build_image_part(url)
            if part:
                parts.append(part)

        if parts:
            contents.append(Content(role="user", parts=parts))

        return contents

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _build_image_part(
        self,
        url: str
    ) -> Optional[Part]:
        """
        Convert an image URL to a Gemini Part.
        Tries URI first; falls back to bytes via GCS if needed.

        Args:
            url: The image URL to convert.

        Returns:
            Optional[Part]: The corresponding Gemini Part or None if conversion fails.
        """

        if not isinstance(url, str) or not url.startswith("http"):
            return None

        mime_type = self._get_mime_type(url)

        try:
            return Part.from_uri(file_uri=url, mime_type=mime_type)
        except Exception as e:
            pass

        if not self.gcs:
            return None

        try:
            filename = url.split("/")[-1].split("?")[0]
            gcs_path = f"{self.config.GCS_IMG_PATH}/{filename}"

            if not self.gcs.exists(gcs_path):
                self.gcs.upload_from_url(
                    url=url,
                    gcs_path=self.config.GCS_IMG_PATH,
                    name=filename,
                )

            image_bytes = self.gcs.download_bytes(gcs_path)
            mime_type = self._get_mime_type_from_bytes(image_bytes)

            return Part.from_bytes(data=image_bytes, mime_type=mime_type)

        except Exception as e:
            return None
