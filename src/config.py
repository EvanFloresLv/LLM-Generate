# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
import os
from pathlib import Path
from typing import Any, Optional, Type

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------
from dotenv import load_dotenv


class Config:
    """
    Runtime configuration loader.
    Loads .env at initialization and exposes typed config values.
    """

    def __init__(self, env_path: Optional[str] = None):
        """
        Initializes the Config instance.

        Args:
            env_path: Optional path to the .env file.

        Returns:
            None
        """
        self._ENV_FILE = (
            Path(env_path).resolve()
            if env_path
            else Path(__file__).resolve().parent.parent / ".env"
        )

        if not self._ENV_FILE.exists():
            raise FileNotFoundError(f".env file not found at: {self._ENV_FILE}")

        load_dotenv(self._ENV_FILE, override=True)


        self.LOCATION = self._getenv("LOCATION", str, required=True)
        self.MODEL_NAME = self._getenv("MODEL_NAME", str, required=True)
        self.TEMPERATURE = self._getenv("TEMPERATURE", float, required=True)
        self.TOP_P = self._getenv("TOP_P", float, required=True)
        self.MAX_OUTPUT_TOKENS = self._getenv("MAX_OUTPUT_TOKENS", int, required=True)
        self.PROMPTS_PATH = self._getenv("PROMPTS_PATH", str, required=True)
        self.DEFAULT_LLM_USAGE_PATH = self._getenv("DEFAULT_LLM_USAGE_PATH", str, required=True)


    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    @staticmethod
    def _getenv(
        key: str,
        cast_type: Optional[Type] = None,
        default: Optional[Any] = None,
        required: bool = False,
    ) -> Any:
        """
        Retrieves an environment variable, with optional type casting and default value.

        Args:
            key: The name of the environment variable.
            cast_type: Optional type to cast the value to.
            default: Optional default value if the variable is not found.
            required: Whether the variable is required.

        Returns:
            The value of the environment variable, cast to the specified type, or the default value.
        """
        value = os.getenv(key, default)

        if isinstance(value, str):
            value = value.strip() or None

        if value is None:
            if required:
                raise EnvironmentError(f"Required environment variable '{key}' is missing.")
            return default

        if cast_type:
            if cast_type is bool:
                return str(value).lower() in {"1", "true", "yes", "on"}
            return cast_type(value)

        return value


if __name__ == "__main__":

    config = Config()