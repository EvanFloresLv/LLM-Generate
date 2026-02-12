import asyncio
from logger.logger import Logger
from llm_sdk.providers.sync_registry import ProviderSpec
from llm_sdk import SyncLLM, AsyncLLM
from llm_sdk_provider_gemini import SyncGeminiClient
from llm_sdk.domain.chat import ChatMessage, ChatPart

async def main() -> None:
    sdk = AsyncLLM.default(load_plugins=True, logger=logger)

    resp = await sdk.chat(
        messages=[
            ChatMessage(
                role="user",
                parts=[
                    ChatPart(
                        type="text",
                        text="¿Qué ves en la imágen?"
                    ),
                    ChatPart(
                        type="image_url",
                        uri="https://verdecora.es/blog/wp-content/uploads/2025/06/cuidados-pato-casa.jpg"
                    )
                ]
            )
        ],
        provider="gemini",
        model="gemini-2.5-flash",
    )

    print(resp.content)


def main_sync() -> None:
    sdk = SyncLLM.default(logger=logger)

    sdk.registry.register(ProviderSpec(
        name="gemini",
        factory=lambda: SyncGeminiClient(
            location=sdk.settings.gemini.location,
        ),
        models={"gemini-2.5-flash"},
    ))

    resp = sdk.chat(
        messages=[
            ChatMessage(
                role="user",
                parts=[
                    ChatPart(
                        type="text",
                        text="¿Qué ves en la imágen?"
                    ),
                    ChatPart(
                        type="image_url",
                        uri="https://verdecora.es/blog/wp-content/uploads/2025/06/cuidados-pato-casa.jpg"
                    )
                ]
            )
        ],
        provider="gemini",
        model="gemini-2.5-flash",
    )

    print(resp.content)

Logger().configure()
logger = Logger()

if __name__ == "__main__":
    # main_sync()
    asyncio.run(main())