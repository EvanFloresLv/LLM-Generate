# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------
from importlib.metadata import entry_points

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from llm_sdk.registry import ProviderRegistry
from llm_sdk.settings import SDKSettings


RegisterFn: Callable[[ProviderRegistry, SDKSettings], None]

@dataclass(frozen=True)
class PluginLoadResult:
    """
    Plugin load result.

    Args:
        loaded: Names loaded successfully.
        failed: Names that failed with error message.
    """

    loaded: list[str]
    failed: dict[str, str]


def load_provider_plugins(registry: ProviderRegistry, settings: SDKSettings) -> PluginLoadResult:
    """
    Load provider plugins from entrypoints.

    Entry point group:
        llm_sdk.providers

    Each entrypoint must be a callable:
        register(registry: ProviderRegistry, settings: SDKSettings) -> None

    Args:
        registry: ProviderRegistry
        settings: SDKSettings

    Returns:
        PluginLoadResult
    """

    eps = entry_points().select(group="llm_sdk.providers")

    loaded: list[str] = []
    failed: dict[str, str] = {}

    for ep in eps:
        try:
            fn = ep.load()
            if not callable(fn):
                failed[ep.name] = "entrypoint is not callable"
                continue

            fn(registry, settings)
            loaded.append(ep.name)
        except Exception as e:
            failed[ep.name] = repr(e)

    return PluginLoadResult(loaded=loaded, failed=failed)