# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------


class GeminiSettings(BaseSettings):
    """
    Settings for Gemini provider.

    Loaded from environment variables using prefix: LLM_SDK_GEMINI_
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_SDK_GEMINI_",
        extra="ignore"
    )

    scopes: list[str] | None = Field(default=None)

    location: str = Field(default="us-central1")
