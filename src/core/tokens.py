# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
import os
import json
from typing import Any, Optional, Dict
from datetime import datetime
from threading import Lock
from tempfile import NamedTemporaryFile


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

    def __new__(cls, config: Optional[Any] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._config = config
                cls._instance._initialized = False
        return cls._instance


    def __init__(self, config: Optional[Any] = None):
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

    def _init_cache(self):
        self.cache: Dict[str, Any] = {
            "model": self._model,
            "calls": 0,
            "total_tokens": {
                "output_tokens": 0,
                "image_tokens": 0,
                "text_tokens": 0,
                "thought_tokens": 0,
            },
            "history": [],
        }

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def add_usage(self, token_count: int, token_type: str):
        if token_type not in self.cache["total_tokens"]:
            raise ValueError(f"Unknown token type: {token_type}")

        with self._data_lock:
            self.cache["total_tokens"][token_type] += int(token_count)
            self.cache["history"].append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "token_type": token_type,
                    "token_count": int(token_count),
                }
            )


    def increment_call(self):
        with self._data_lock:
            self.cache["calls"] += 1

    # -----------------------------------------------------------------
    # Persistence (atomic write)
    # -----------------------------------------------------------------

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        with NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            json.dump(self.cache, tmp, indent=2)
            temp_path = tmp.name

        os.replace(temp_path, self.path)


    def reset(self):
        with self._data_lock:
            self._init_cache()
