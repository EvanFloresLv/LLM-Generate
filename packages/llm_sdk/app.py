from logger.logger import Logger, LoggingSettings

from llm_sdk import SyncLLM
from llm_sdk.providers.sync_registry import ProviderSpec
from llm_sdk.providers.noop_client import NoopLLMClient

Logger().configure()
logger = Logger()

def main_sync() -> None:
    sdk = SyncLLM.default(logger=logger)

    sdk.registry.register(ProviderSpec(
        name="noop",
        factory=lambda: NoopLLMClient(),
        models={"noop"},
    ))

    resp = sdk.chat(
        messages=[{"model", "Eres un profesor"}, ("user", "Hola, ¿Cómo estás?")],
        provider="noop",
        model="noop",
    )

    print(resp.content)


if __name__ == "__main__":
    main_sync()