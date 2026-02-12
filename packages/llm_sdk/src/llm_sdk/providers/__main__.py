# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

# ---------------------------------------------------------------------
# Internal application imports
# ---------------------------------------------------------------------
from __future__ import annotations
from llm_sdk.providers.async_registry import ProviderRegistry


def main() -> None:
    reg = ProviderRegistry()
    reg.load_plugins()

    print("Installed providers:")
    for name in reg.available():
        print(f" - {name}")


if __name__ == "__main__":
    main()
