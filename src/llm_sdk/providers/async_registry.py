# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Protocol

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.providers.async_base import AsyncBaseLLMClient
from llm_sdk.settings import SDKSettings


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    name: str
    supports_chat: bool
    supports_embeddings: bool
    supports_streaming: bool
    is_async: bool


class ProviderFactory(Protocol):
    """
    Provider plugin contract.

    Every provider plugin must expose a factory implementing this protocol.
    """

    def spec(self) -> ProviderSpec: ...

    def create(self, settings: SDKSettings) -> AsyncBaseLLMClient: ...


class ProviderRegistry:
    """
    Registry that loads provider plugins using Python entry points.
    """

    ENTRYPOINT_GROUP = "llm_sdk.providers"


    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}


    def load_plugins(self) -> None:
        """
        Load all installed provider plugins via entry points.
        """
        eps = entry_points()
        group = eps.select(group=self.ENTRYPOINT_GROUP)

        for ep in group:
            factory_cls = ep.load()
            factory: ProviderFactory = factory_cls()
            spec = factory.spec()
            self._factories[spec.name] = factory


    def get(self, name: str) -> ProviderFactory:
        """
        Get provider factory by name.

        Args:
            name: Provider name.

        Returns:
            ProviderFactory.
        """
        try:
            return self._factories[name]
        except KeyError as e:
            available = ", ".join(sorted(self._factories.keys()))
            raise KeyError(f"Unknown provider '{name}'. Available: {available}") from e


    def available(self) -> list[str]:
        """
        Return installed provider names.
        """
        return sorted(self._factories.keys())
