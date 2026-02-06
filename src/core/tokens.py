# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
import os
import json
from dataclasses import asdict
from typing import Any, Optional
from datetime import datetime, timezone
from threading import Lock
from tempfile import NamedTemporaryFile

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from src.schemas.token_schema import TokenUsage, HistoryItem


class TokenTracker:
    """
    Singleton tracker for LLM token usage per pipeline run.
    Thread-safe and crash-safe persistence.
    """

    _instance: "TokenTracker | None" = None
    _lock = Lock()

    # -----------------------------------------------------------------
    # Singleton construction
    # -----------------------------------------------------------------

    def __new__(
        cls,
        config: Optional[Any] = None
    ):
        """
        Creates a new instance of the TokenTracker.

        Args:
            config: The configuration object containing model and path information.

        Returns:
            None
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance


    def __init__(
        self,
        config: Optional[Any] = None
    ):
        """
        Initializes the token tracker.

        Args:
            config: The configuration object containing model and path information.

        Returns:
            None
        """
        if self._initialized:
            return

        if config is None:
            raise ValueError("Config must be provided on first initialization")

        self._model = getattr(config, "MODEL_NAME", "unknown")
        self._default_path = getattr(
            config,
            "DEFAULT_LLM_USAGE_PATH",
            "./llm_usage.json",
        )
        self.path = self._default_path

        self._data_lock = Lock()
        self._init_cache()
        self._initialized = True

    # -----------------------------------------------------------------
    # Internal state
    # -----------------------------------------------------------------

    def _init_cache(self) -> None:
        """
        Initializes the cache for token usage.
        """
        self.cache: TokenUsage = TokenUsage(model=self._model)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def add_usage(
        self,
        token_count: int,
        token_type: str
    ) -> None:
        """
        Adds token usage to the tracker.

        Args:
            token_count: The number of tokens used.
            token_type: The type of tokens used (e.g., "input", "output").

        Returns:
            None
        """
        token_count = int(token_count)

        if not hasattr(self.cache.tokens, token_type):
            raise ValueError(f"Unknown token type: {token_type}")

        with self._data_lock:
            # Update per-type
            current = getattr(self.cache.tokens, token_type)
            setattr(self.cache.tokens, token_type, current + token_count)

            # Update total
            self.cache.total_tokens += token_count

            # Add history item
            self.cache.history.append(
                HistoryItem(
                    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    token_type=token_type,
                    token_count=token_count,
                )
            )


    def get_usage(self) -> dict[str, Any]:
        """
        Returns the current token usage statistics.
        """
        with self._data_lock:
            return asdict(self.cache)


    def increment_call(self) -> None:
        """
        Increments the call count for the current pipeline run.
        """
        with self._data_lock:
            self.cache.calls += 1

    # -----------------------------------------------------------------
    # Persistence (atomic write)
    # -----------------------------------------------------------------

    def save(self) -> None:
        """
        Saves the current token usage data to a file.
        """
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        payload = asdict(self.cache)

        with NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            json.dump(payload, tmp, indent=2)
            temp_path = tmp.name

        os.replace(temp_path, self.path)

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------

    def reset(self) -> TokenUsage:
        """
        Resets cache and returns a snapshot of the previous usage.
        """
        with self._data_lock:
            snapshot = self.cache
            self._init_cache()
            return snapshot
