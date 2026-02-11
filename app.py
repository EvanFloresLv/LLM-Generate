import asyncio
from llm_sdk.async_sdk import AsyncLLM


async def main() -> None:
    sdk = AsyncLLM.default()
    sdk.registry.load_plugins()

    resp = await sdk.chat(
        messages=[("user", "Hola, ¿Cómo estás?")],
        provider="gemini",
        model="gemini-2.5-flash",
    )

    print(resp.content)


if __name__ == "__main__":
    asyncio.run(main())