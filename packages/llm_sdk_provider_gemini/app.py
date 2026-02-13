import asyncio
from dataclasses import replace

from logger.logger import Logger
from google.genai.types import GenerateContentConfig

from llm_sdk.providers.sync_registry import ProviderSpec
from llm_sdk import SyncLLM, AsyncLLM
from llm_sdk_provider_gemini import SyncGeminiClient, AsyncGeminiLLMClient
from llm_sdk.domain.chat import ChatMessage, ChatPart

from llm_sdk.retries import RetryPolicy

async def main() -> None:
    sdk = AsyncLLM.default(load_plugins=True, logger=logger)

    retry_policy = RetryPolicy(
        max_attempts=3,
        base_delay_s=2,
    )

    sdk = replace(sdk, retry_policy=retry_policy)

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        },
        "required": ["summary", "sentiment"],
    }

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
        output_schema=schema,
    )

    text_out: list[str] = []
    last_usage = None

    async for ev in sdk.stream_chat(
        messages=[
            ChatMessage(
                role="user",
                parts=[
                    ChatPart(type="text", text="¿Qué ves en la imágen?"),
                    ChatPart(
                        type="image_url",
                        uri="https://verdecora.es/blog/wp-content/uploads/2025/06/cuidados-pato-casa.jpg",
                    ),
                ],
            )
        ],
        provider="gemini",
        model="gemini-2.5-flash",
        output_schema=schema,
    ):
        if ev.delta:
            print(ev.delta, end="", flush=True)
            text_out.append(ev.delta)
        if ev.usage:
            last_usage = ev.usage
        if ev.done:
            break

    print("\n\n[done]")
    print("usage:", last_usage)
    full_text = "".join(text_out)
    print("full text:", full_text)


def main_sync() -> None:
    sdk = SyncLLM.default(logger=logger)

    sdk.registry.register(ProviderSpec(
        name="gemini",
        factory=lambda: SyncGeminiClient(
            location=sdk.settings.gemini.location,
        ),
        models={"gemini-2.5-flash"},
    ))

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        },
        "required": ["summary", "sentiment"],
    }

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
        output_schema=schema,
    )

    print(resp.content)

    # text_out: list[str] = []
    # last_usage = None
    # for ev in sdk.stream_chat(
    #     messages=[
    #         ChatMessage(
    #             role="user",
    #             parts=[
    #                 ChatPart(type="text", text="¿Qué ves en la imágen?"),
    #                 ChatPart(
    #                     type="image_url",
    #                     uri="https://verdecora.es/blog/wp-content/uploads/2025/06/cuidados-pato-casa.jpg",
    #                 ),
    #             ],
    #         )
    #     ],
    #     provider="gemini",
    #     model="gemini-2.5-flash",
    #     output_schema=schema,
    # ):
    #     if ev.delta:
    #         text_out.append(ev.delta)
    #     if ev.usage:
    #         last_usage = ev.usage
    #     if ev.done:
    #         break

    # print("\n\n[done]")
    # print("usage:", last_usage)
    # full_text = "".join(text_out)
    # print("full text:", full_text)


Logger().configure()
logger = Logger()


if __name__ == "__main__":
    main_sync()
    # asyncio.run(main())