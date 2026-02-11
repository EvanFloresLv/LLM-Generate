import asyncio

from llm_sdk.sync.sdk import LLM
from llm_sdk.sync.providers.gemini.client import GeminiLLMClient
from llm_sdk.sync.providers.registry import ProviderSpec

from llm_sdk._async.async_sdk import AsyncLLM

async def main() -> None:
    sdk = AsyncLLM.default()
    sdk.registry.load_plugins()

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
    sdk = LLM.default()

    sdk.registry.register(ProviderSpec(
        name="gemini",
        factory=lambda: GeminiLLMClient(
            location=sdk.settings.gemini.location,
        ),
        models={"gemini-2.5-flash"},
    ))

    resp = sdk.chat(
        messages=[{"model", "Eres un profesor"}, ("user", "Hola, ¿Cómo estás?")],
        provider="gemini",
        model="gemini-2.5-flash",
    )

    print(resp.content)


if __name__ == "__main__":
    # main_sync()
    asyncio.run(main())