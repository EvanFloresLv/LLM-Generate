from llm_sdk.sdk import LLM
from llm_sdk.async_sdk import AsyncSDK

from llm_sdk.registry import ProviderSpec

from llm_sdk.providers.noop_client import NoopLLMClient
from llm_sdk.providers.gemini_client import GeminiLLMClient

from llm_sdk.providers.async_noop_client import AsyncNoopLLMClient
from llm_sdk.providers.async_gemini_client import AsyncGeminiLLMClient

from llm_sdk.timeouts import TimeoutConfig
from llm_sdk.settings import load_settings

settings = load_settings()

sdk = LLM.default()
sdk.settings = settings

sdk.registry.register(
    ProviderSpec(
        name="gemini",
        factory=lambda: GeminiLLMClient(location="us-central1", timeouts=TimeoutConfig()),
        models={"gemini-2.5-flash", "text-multilingual-embedding-002"},
    )
)


# resp = sdk.chat(messages=[("user", "Hola, ¿Cómo estás?")], provider="gemini", model="gemini-2.5-flash")
# print(resp.content)

# for ev in sdk.stream_chat(messages=[("user", "Hola, ¿Cómo estás?"), ("user", "Explica AI")], provider="gemini", model="gemini-2.5-flash"):
#     print(ev.delta, end="", flush=True)
#     if ev.done:
#         print("\n[Stream complete]")

# resp = sdk.embed(input=["Hola, ¿Cómo estás?"], provider="gemini", model="text-multilingual-embedding-002")
# print(resp.vectors)