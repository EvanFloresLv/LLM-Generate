# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
import os
import json
import functools
from datetime import datetime
from threading import Lock
from typing import Optional

# ---------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------
from google.genai import types

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from src.config import Config


class TokenTracker:
    """
    Singleton tracker for LLM token usage per pipeline run.
    Tracks total tokens, call count, and usage history.
    """
    _instance = None
    _lock = Lock()
    _default_path = Config().DEFAULT_LLM_USAGE_PATH
    _model = Config().MODEL_NAME

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_cache()
            return cls._instance


    def __init__(self):
        # Only initialize once
        if not hasattr(self, "cache"):
            self._init_cache()
        self.path = self._default_path


    def _init_cache(self):
        self.cache = {
            "total_tokens": {
                "output_tokens": 0,
                "image_tokens": 0,
                "text_tokens": 0,
                "thought_tokens": 0,
            },
            "calls": 0,
            "model": self._model,
            "history": []
        }


    def add_usage(self, token_count: int, token_type: str, model: Optional[str] = None):
        """
        Add token usage to the tracker.
        Args:
            token_count (int): Number of tokens used.
            token_type (str): Type of token (output_tokens, image_tokens, text_tokens, thought_tokens).
            model (Optional[str]): Model name.
        """
        if token_type not in self.cache["total_tokens"]:
            raise ValueError(f"Unknown token type: {token_type}")
        self.cache["total_tokens"][token_type] += int(token_count)
        self.cache["calls"] += 1
        self.cache["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "token_type": token_type,
            "token_count": int(token_count),
        })


    def save(self):
        """
        Save the current token usage cache to a JSON file.
        Args:
            path (Optional[str]): Path to save the file. Uses default if None.
        """
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)


    def reset(self):
        """Reset the token usage cache to initial state."""
        self._init_cache()


def save_llm_usage(model: Optional[str] = None):
    """
    Decorator to save Gemini LLM token usage to the singleton tracker.
    Assumes the wrapped function returns a response with .usage_metadata.
    """
    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            response = func(*args, **kwargs)
            if not hasattr(response, "text") or response.text is None:
                return response

            usage_metadata = getattr(response, "usage_metadata", None)
            if not usage_metadata:
                return response

            tracker = TokenTracker()

            # Output tokens (candidates)
            candidates_token_count = getattr(usage_metadata, "candidates_token_count", None)
            if candidates_token_count:
                tracker.add_usage(candidates_token_count, "output_tokens", model)

            # Prompt tokens by modality
            prompt_tokens_details = getattr(usage_metadata, "prompt_tokens_details", None)
            if prompt_tokens_details:
                for part in prompt_tokens_details:
                    modality = getattr(part, "modality", None)
                    token_count = getattr(part, "token_count", None)
                    if modality == types.MediaModality.IMAGE:
                        tracker.add_usage(token_count, "image_tokens", model)
                    elif modality == types.MediaModality.TEXT:
                        tracker.add_usage(token_count, "text_tokens", model)

            # Thought tokens
            thoughts_token_count = getattr(usage_metadata, "thoughts_token_count", None)
            if thoughts_token_count:
                tracker.add_usage(thoughts_token_count, "thought_tokens", model)

            tracker.save()
            return response
        return wrapper
    return decorator