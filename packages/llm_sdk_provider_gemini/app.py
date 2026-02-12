import asyncio

from logger.logger import Logger

from llm_sdk.sync_sdk import LLM
from llm_sdk.providers.sync_registry import ProviderSpec

from llm_sdk import SyncLLM, AsyncLLM

from llm_sdk_provider_gemini import SyncGeminiClient

async def main() -> None:
    sdk = AsyncLLM.default(load_plugins=True, logger=logger)

    resp = await sdk.chat(
        messages=[("model", "Eres un profesor de programación"), ("user", "Hola, ¿Cómo estás?")],
        provider="gemini",
        model="gemini-2.5-flash",
    )

    emb = await sdk.embed(
        input=["Hola, ¿Cómo estás?"],
        provider="gemini",
        model="text-multilingual-embedding-002",
    )

    print(resp.content)
    print(emb.vectors)


def main_sync() -> None:
    sdk = LLM.default(logger=logger)

    sdk.registry.register(ProviderSpec(
        name="gemini",
        factory=lambda: SyncGeminiClient(
            location=sdk.settings.gemini.location,
        ),
        models={"gemini-2.5-flash"},
    ))

    resp = sdk.chat(
        messages=[("model", "Eres un profesor"), ("user", "Hola, ¿Cómo estás?")],
        provider="gemini",
        model="gemini-2.5-flash",
    )

    print(resp.content)

Logger().configure()
logger = Logger()

if __name__ == "__main__":
    main_sync()
    asyncio.run(main())