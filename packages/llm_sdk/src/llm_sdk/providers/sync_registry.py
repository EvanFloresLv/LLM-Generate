# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.exceptions import ModelNotFoundError, ProviderNotFoundError
from llm_sdk.providers.sync_base import BaseLLMClient


ProviderFactory = Callable[[], BaseLLMClient]


@dataclass(slots=True)
class ProviderSpec:
    """
    Provider spec stored in registry.

    Args:
        name: Provider name.
        factory: Factory that builds the provider client.
        models: Allowed models for this provider.
    """

    name: str
    factory: ProviderFactory
    models: set[str]


class ProviderRegistry:
    """
    Registry of providers.

    This is the key to Open/Closed:
    - core never imports provider modules directly
    - providers register themselves externally
    """

    def __init__(self) -> None:
        self._providers: dict[str, ProviderSpec] = {}


    def register(self, spec: ProviderSpec) -> None:
        """
        Register a provider spec.

        Args:
            spec: ProviderSpec
        """
        self._providers[spec.name] = spec


    def get(self, name: str) -> ProviderSpec:
        """
        Get provider spec by name.

        Args:
            name: Provider name.

        Returns:
            ProviderSpec

        Raises:
            ProviderNotFoundError
        """
        if name not in self._providers:
            raise ProviderNotFoundError(f"Provider not registered: {name}")
        return self._providers[name]


    def resolve_model(self, provider: str, model: str) -> str:
        """
        Validate that model exists for provider.

        Args:
            provider: Provider name.
            model: Model name.

        Returns:
            model (unchanged)

        Raises:
            ModelNotFoundError
        """
        spec = self.get(provider)

        if model not in spec.models:
            raise ModelNotFoundError(f"Model '{model}' not found for provider '{provider}'")

        return model


    def list_providers(self) -> list[str]:
        """
        List registered providers.

        Returns:
            list[str]
        """
        return sorted(self._providers.keys())
