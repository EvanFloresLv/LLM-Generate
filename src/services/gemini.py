# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
import copy
from types import SimpleNamespace
from threading import Lock
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------
from google import genai
from google.auth import default
from google.genai.types import (
    Content,
    GenerateContentConfig,
    SafetySetting,
    Tool,
)

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from src.services.base import LLMProvider
from src.decorators.usage import token_usage


class Gemini(LLMProvider):
    """Google Gemini implementation of the LLMProvider."""

    IMAGE_MIME_MAP: Dict[str, str] = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }

    _instance: "Gemini | None" = None
    _lock = Lock()

    def __new__(
        cls,
        config: Any = None
    ):
        """
        Singleton instance creation.

        Args:
            config: Configuration for the Gemini instance.

        Returns:
            Gemini: The singleton instance of the Gemini class.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance


    def __init__(
        self,
        config: Any
    ) -> None:
        """
        Initialize the Gemini instance.

        Args:
            config: Configuration for the Gemini instance.

        Returns:
            None
        """
        if self._initialized:
            return  # Prevent re-init in singleton

        if isinstance(config, dict):
            self._config = SimpleNamespace(**config)
        else:
            self._config = config

        self._model_name = getattr(self._config, "MODEL_NAME", "gemini-2.5-flash")

        self._credentials, self._project_id = default(
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/generative-language",
            ]
        )

        self._client: Optional[genai.Client] = None
        self._tools: List[Tool] = []
        self._provider_name = "gemini"

        self._initialized = True

    # -----------------------------------------------------------------
    # Client + config
    # -----------------------------------------------------------------

    def _create_safety_settings(self) -> List[SafetySetting]:
        """
        Create safety settings for the Gemini instance.

        Args:
            None

        Returns:
            List[SafetySetting]: A list of safety settings for the Gemini instance.
        """
        categories = [
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_HARASSMENT",
        ]
        return [SafetySetting(category=cat, threshold="OFF") for cat in categories]


    def initialize_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                vertexai=True,
                project=self._project_id,
                location=self._config.LOCATION,
                credentials=self._credentials,
            )
        return self._client

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _remove_images(self, contents: List[Content]) -> List[Content]:
        """
        Removes image parts from Gemini contents.
        Useful for fallback when URLs are rejected.

        Args:
            contents: A list of Content objects to process.

        Returns:
            List[Content]: A list of Content objects without image parts.
        """
        filtered_contents: List[Content] = []

        for content in contents:
            content_copy = copy.deepcopy(content)
            new_parts = []

            for part in getattr(content_copy, "parts", []) or []:

                # Dict-like parts
                if isinstance(part, dict):
                    if any(k in part for k in ("url", "image_url", "image", "image_data", "file_data", "inline_data")):
                        continue
                    new_parts.append(part)
                    continue

                # Object-like parts
                is_image_part = any(
                    hasattr(part, attr)
                    for attr in ("url", "image_url", "image", "image_data", "file_data", "inline_data")
                )

                if not is_image_part:
                    new_parts.append(part)

            if new_parts:
                content_copy.parts = new_parts
                filtered_contents.append(content_copy)

        return filtered_contents

    # -----------------------------------------------------------------
    # Core request
    # -----------------------------------------------------------------

    @token_usage(provider="gemini")
    def _send_request(self, contents: List[Content], **kwargs):
        """
        Sends a request to the Gemini model and returns the response.

        IMPORTANT:
        - This function must return either:
            response
          OR
            (response, input_tokens)
        depending on how your decorator is implemented.

        In your current TokenTracker approach, returning just response is enough,
        because usage_metadata already contains token usage.

        Args:
            contents: A list of Content objects to process.

        Returns:
            Union[Response, Tuple[Response, int]]: The response from the Gemini model, and optionally the input token count.
        """

        client = self.initialize_client()

        config = GenerateContentConfig(
            temperature=kwargs.get("temperature", self._config.TEMPERATURE),
            top_p=kwargs.get("top_p", self._config.TOP_P),
            max_output_tokens=kwargs.get("max_output_tokens", self._config.MAX_OUTPUT_TOKENS),
            safety_settings=self._create_safety_settings(),
            response_modalities=kwargs.get("response_modalities", ["TEXT"]),
            response_mime_type=kwargs.get("mime_type", "application/json"),
            response_schema=kwargs.get("response_schema", None),
        )

        response = client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )

        if not response or not getattr(response, "text", None):
            raise ValueError("Empty response from Gemini model")

        return response

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def generate_response(
        self,
        contents: List[Content],
        **kwargs
    ) -> str:
        """
        Generates a response from the Gemini model.

        Args:
            contents: A list of Content objects to process.

        Returns:
            str: The generated response text.
        """

        if not contents:
            raise ValueError("Contents must be a non-empty list")

        try:
            response = self._send_request(contents, **kwargs)
            return response.text

        except Exception as e:
            error_msg = str(e)

            if "URL_REJECTED" in error_msg or "INVALID_ARGUMENT" in error_msg:
                filtered_contents = self._remove_images(contents)

                if not filtered_contents:
                    raise RuntimeError("All contents were removed after filtering images.")

                response = self._send_request(filtered_contents, **kwargs)
                return response.text

            raise RuntimeError(f"Unexpected error occurred: {error_msg}")


if __name__ == "__main__":

    from src.core.tokens import TokenTracker
    from src.formatter.gemini import GeminiPromptFormatter

    class ConfigDemo:
        LOCATION = "us-central1"
        MODEL_NAME = "gemini-2.5-flash"
        TEMPERATURE = 0.4
        TOP_P = 0.5
        MAX_OUTPUT_TOKENS = 8192
        PROMPTS_PATH = "./src/mocks/template_prompt.yaml"
        DEFAULT_LLM_USAGE_PATH = "./src/data/cache/cache.json"

    config = ConfigDemo()
    gemini = Gemini(config)

    TokenTracker(config=config)  # Initialize TokenTracker singleton
    formatter = GeminiPromptFormatter(config=config)

    response_1 = gemini.generate_response(
        contents=formatter.format(
            prompt={"user": "Hola, ¿cómo estás?"}, image_urls=[]
        )
    )
    token_usage = TokenTracker().reset()
    response_2 = gemini.generate_response(
        contents=formatter.format(
            prompt={"user": "¿Qué es IA?"}, image_urls=[]
        )
    )
    print(response_1)
    print(response_2)