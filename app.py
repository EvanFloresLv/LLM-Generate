import asyncio

from llm_sdk.sync.sdk import LLM
from llm_sdk.sync.providers.gemini.client import GeminiLLMClient
from llm_sdk.sync.providers.registry import ProviderSpec

from llm_sdk._async.async_sdk import AsyncLLM

async def main() -> None:
    sdk = AsyncLLM.default()
    sdk.registry.load_plugins()

    resp = await sdk.chat(
        messages=[("user", "Hola, ¿Cómo estás?")],
        provider="gemini",
        model="gemini-2.5-flash",
    )

    print(resp.content)


def main_sync() -> None:
    sdk = LLM.default()

    sdk.registry.register(ProviderSpec(
        name="gemini",
        factory=lambda: GeminiLLMClient(
            location=sdk.settings.gemini.location,
        ),
        models={"gemini-2.5-flash"},
    ))

    resp = sdk.chat(
        messages=[("user", "Hola, ¿Cómo estás?")],
        provider="gemini",
        model="gemini-2.5-flash",
    )

    print(resp.content)


if __name__ == "__main__":
    main_sync()
    asyncio.run(main())