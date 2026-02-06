# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class Tokens:
    input: int = 0
    output: int = 0
    image: int = 0
    text: int = 0
    thought: int = 0


@dataclass
class HistoryItem:
    timestamp: str
    token_type: str
    token_count: int


@dataclass
class TokenUsage:
    model: str
    calls: int = 0
    total_tokens: int = 0
    tokens: Tokens = field(default_factory=Tokens)
    history: list[HistoryItem] = field(default_factory=list)
