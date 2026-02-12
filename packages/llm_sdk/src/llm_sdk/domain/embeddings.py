# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """
    Embeddings request.

    Args:
        model: Embedding model.
        input: Input texts.
        metadata: Optional metadata.
    """

    model: str
    input: list[str]
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """
    Embeddings response.

    Args:
        model: Model used.
        vectors: Embeddings vectors aligned with input.
        raw: Provider raw payload.
    """

    model: str
    vectors: list[list[float]]
    raw: dict[str, Any] | None = None
