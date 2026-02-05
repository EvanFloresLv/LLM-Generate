# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
import copy
from types import SimpleNamespace
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
    Tool
)

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------

from src.services.base import LLMProvider
from src.core.tokens import save_llm_usage


class Gemini(LLMProvider):
    """Google Gemini implementation of the LLMProvider."""

    IMAGE_MIME_MAP: Dict[str, str] = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }

    def __init__(self, config: Any) -> None:
        """
        Initialize Gemini provider with credentials and configuration.

        Args:
            config (Any): Configuration object containing credentials paths and model settings.

        Returns:
            None
        """

        if isinstance(config, dict):
            self._config = SimpleNamespace(**config)
        else:
            self._config = config

        self._credentials, self._project_id = default(
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/generative-language",
            ]
        )

        self._client: Optional[genai.Client] = None
        self._model_config: Optional[GenerateContentConfig] = None
        self._tools: List[Tool] = []
        self._provider_name = "gemini"


    def _create_safety_settings(self) -> List[SafetySetting]:
        """
        Define Gemini safety settings with relaxed thresholds.

        Args:
            None

        Returns:
            List[SafetySetting]: List of safety settings with thresholds OFF.
        """

        categories = [
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_HARASSMENT",
        ]
        return [SafetySetting(category=cat, threshold="OFF") for cat in categories]


    def initialize_client(self) -> genai.Client:
        """
        Initialize Gemini client with credentials.

        Args:
            None

        Returns:
            genai.Client: Initialized Gemini client.
        """

        if self._client is None:
            self._client = genai.Client(
                vertexai=True,
                project=self._project_id,
                location=self._config.LOCATION,
                credentials=self._credentials,
            )
        return self._client


    def _remove_images(self, contents: List[Content]) -> List[Content]:
        """
        Remove image contents from the list.

        Args:
            contents (List[Content]): The list of content to filter.

        Returns:
            List[Content]: The filtered list of content without images.
        """
        filtered_contents = []

        for content in contents:
            content_copy = copy.deepcopy(content)
            parts = []

            for part in getattr(content_copy, "parts", []):
                if isinstance(part, dict):
                    if any(k in part for k in ["url", "image_url", "image", "image_data"]):
                        continue
                elif hasattr(part, ("url", "image_url", "image", "image_data")):
                    continue
                parts.append(part)

            if parts:
                content_copy.parts = parts
                filtered_contents.append(content_copy)

        return filtered_contents


    @save_llm_usage(model="gemini-2.5-pro")
    def _send_request(self, contents: List[Content], **kwargs):
        """
        Send a request to the Gemini API with the given content and parameters.

        Args:
            contents (List[Content]): The list of content to send in the request.
            **kwargs: Additional keyword arguments to customize the request.

        Returns:
            Tuple[str, Optional[UsageMetadata]]: The response from the Gemini API and its usage metadata.
        """
        client = self.initialize_client()

        config = GenerateContentConfig(
            temperature=kwargs.get("temperature", self._config.TEMPERATURE),
            top_p=kwargs.get("top_p", self._config.TOP_P),
            max_output_tokens=kwargs.get("max_output_tokens", self._config.MAX_OUTPUT_TOKENS),
            safety_settings=self._create_safety_settings(),
            response_modalities=kwargs.get("response_modalities", ["TEXT"]),
            response_mime_type=kwargs.get("mime_type", "application/json"),
            response_schema=kwargs.get("response_schema"),
        )

        response = client.models.generate_content(
            model=self._config.MODEL_NAME,
            contents=contents,
            config=config,
        )

        if not response or not response.text:
            raise ValueError("Empty response from Gemini model")

        return response


    def generate_response(self, contents: List[Content], **kwargs) -> str:
        """
        Generates a response from the Gemini API for the given content.

        Args:
            contents (List[Content]): The list of content to send in the request.
            **kwargs: Additional keyword arguments to customize the request.

        Returns:
            str: The response from the Gemini API.
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

                response = self._send_request(filtered_contents)
                return response.text

            raise

# --------------------------------------------------------------------------------------------------------------------#
#                                     Standalone execution for testing purposes                                       #
# --------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    from src.core.formatter import GeminiPromptFormatter

    class ConfigDemo:
        PATH_DATA = "src/data"
        PROJECT_ID = "crp-dev-dig-plantillase"
        LOCATION = "us-central1"
        MODEL_NAME = "gemini-2.5-flash"
        TEMPERATURE = 0.4
        TOP_P = 0.5
        MAX_OUTPUT_TOKENS = 8192
        PROMPTS_PATH = "./src/prompts/prompt_attribute_validation.yaml"

        PROJECT_ID_BQ_AT = "crp-pro-dig-plantillase"
        SAP_DATA_PATH = "src/data/Sap.xlsx"

    config = ConfigDemo()
    gemini = Gemini(config)
    formatter = GeminiPromptFormatter()

    response_1 = gemini.generate_response(
        contents=formatter.format(
            prompt={
                "user": "Hola, ¿cómo estás?"
            },
            image_urls=[]
        )
    )

    from src.core.token import TokenTracker

    token_usage = TokenTracker().reset()

    response_2 = gemini.generate_response(
        contents=formatter.format(
            prompt={
                "user": "¿Qué es IA?"
            },
            image_urls=[]
        )
    )

    print(response_1)
    print(response_2)