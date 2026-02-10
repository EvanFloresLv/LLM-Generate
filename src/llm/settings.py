# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------
from google.auth import default
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any
from pydantic import PrivateAttr


class GeminiCredentials(BaseSettings):
    _credentials: Any = PrivateAttr()
    _project: Any = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._credentials, self._project = default()


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="_",
        env_nested_max_split=2,
        env_prefix="LLM_",
        extra="ignore",
    )

    LOCATION: str
    MODEL_NAME: str
    TEMPERATURE: float
    TOP_P: float
    MAX_OUTPUT_TOKENS: int
    PROMPTS_PATH: str
    DEFAULT_LLM_USAGE_PATH: str