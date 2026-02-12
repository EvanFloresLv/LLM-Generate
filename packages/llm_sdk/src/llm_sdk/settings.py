# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.retries import RetryPolicy
from llm_sdk.timeouts import TimeoutConfig


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


class SDKSettings(BaseSettings):
    """
    Global SDK settings loaded from environment variables.

    Env prefix: LLM_SDK_

    Examples:
        LLM_SDK_ENV=prod
        LLM_SDK_DEFAULT_PROVIDER=openai
        LLM_SDK_DEFAULT_MODEL=gpt-4o-mini
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LLM_SDK_",
        extra="ignore",
    )

    gemini: GeminiSettings = GeminiSettings()

    env: Literal["dev", "prod"] = "dev"

    default_provider: str = "noop"
    default_model: str = "noop-model"

    retries: RetryPolicy = RetryPolicy()
    timeouts: TimeoutConfig = TimeoutConfig()


def load_settings(**kwargs) -> SDKSettings:
    """
    Load SDK settings.

    Returns:
        SDKSettings
    """
    return SDKSettings(**kwargs)
