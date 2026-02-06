# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from typing import Any

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from src.schemas.token_schema import TokenUsage


class GeminiUsageExtractor:
    def extract(self, response: Any, model: str) -> TokenUsage:
        usage = TokenUsage(model=model)

        usage_metadata = getattr(response, "usage_metadata", None)
        if not usage_metadata:
            return usage

        usage.tokens.output = int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
        usage.tokens.thought = int(getattr(usage_metadata, "thoughts_token_count", 0) or 0)

        prompt_details = getattr(usage_metadata, "prompt_tokens_details", None) or []
        for part in prompt_details:
            token_count = int(getattr(part, "token_count", 0) or 0)
            modality = str(getattr(part, "modality", "")).upper()

            if "IMAGE" in modality:
                usage.tokens.image += token_count
            elif "TEXT" in modality:
                usage.tokens.text += token_count
            else:
                usage.tokens.input += token_count

        usage.total_tokens = (
            usage.tokens.input
            + usage.tokens.output
            + usage.tokens.image
            + usage.tokens.text
            + usage.tokens.thought
        )

        return usage
