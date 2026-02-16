# llm-sdk-provider-gemini

Gemini provider plugin for `llm-sdk`, implemented using `google-genai` (Vertex AI).
This package installs a provider plugin via Python entry points, enabling Gemini support without modifying the core SDK.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

`llm-sdk-provider-gemini` is a plugin package that adds support for:

- Gemini chat completions
- Gemini embeddings
- Gemini streaming chat

It integrates with the `llm-sdk` core using **Python entry points**, so the provider is automatically discoverable via:

```python
sdk.registry.load_plugins()
```

---

## Architecture

### Plugin mechanism

This package declares an entry point:

```bash
[project.entry-points."llm_sdk.providers"]
gemini = "llm_sdk_provider_gemini.plugin:GeminiProviderFactory"
```

At runtime:

1. ```llm-sdk``` scans installed packages for llm_sdk.providers
2. Loads the GeminiProviderFactory
3. Calls:
    - spec() for metadata
    - create(settings) to build a provider client

---

## Features

- Async Gemini client
- Supports:
    - chat completions
    - embeddings
    - streaming chat
- Uses google-genai (Vertex AI mode)
- Unified SDK error mapping (ProviderError)
- Strong typing
- Clean separation from SDK core

---

## Installation

### Install core SDK

```bash
pip install llm-sdk
```

### Install Gemini provider

```bash
pip install llm-sdk-provider-gemini
```

---

## Usage

### Sync quickstart

```python
from llm_sdk.sync_sdk import SyncSDK
from llm_sdk.retries import RetryPolicy
from llm_sdk.providers.sync_registry import ProviderSpec
from llm_sdk_provider_gemini import SyncGeminiClient
from llm_sdk.domain.chat import ChatMessage, ChatPart

sdk = SyncSDK.default()
sdk.registry.register(ProviderSpec(
    name="gemini",
    factory=lambda: SyncGeminiClient(
        location=sdk.settings.gemini.location,
    ),
    models={
        "gemini-2.5-flash",
        "text-multilingual-embedding-002"
    },
))
```

### Normal chat example

```python
from llm_sdk.sync_sdk import SyncSDK
from llm_sdk.retries import RetryPolicy
from llm_sdk.providers.sync_registry import ProviderSpec
from llm_sdk_provider_gemini import SyncGeminiClient
from llm_sdk.domain.chat import ChatMessage, ChatPart

sdk = SyncSDK.default()
sdk.registry.register(ProviderSpec(
    name="gemini",
    factory=lambda: SyncGeminiClient(
        location=sdk.settings.gemini.location,
    ),
    models={
        "gemini-2.5-flash",
    },
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
```

### Streaming example

```python
from llm_sdk.sync_sdk import SyncSDK
from llm_sdk.retries import RetryPolicy
from llm_sdk.providers.sync_registry import ProviderSpec
from llm_sdk_provider_gemini import SyncGeminiClient
from llm_sdk.domain.chat import ChatMessage, ChatPart

sdk = SyncSDK.default()
sdk.registry.register(ProviderSpec(
    name="gemini",
    factory=lambda: SyncGeminiClient(
        location=sdk.settings.gemini.location,
    ),
    models={
        "gemini-2.5-flash",
        "text-multilingual-embedding-002"
    },
))

text_out: list[str] = []
last_usage = None
for ev in sdk.stream_chat(
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
):
    if ev.delta:
        text_out.append(ev.delta)
    if ev.usage:
        last_usage = ev.usage
    if ev.done:
        break

print("\n\n[done]")
print("usage:", last_usage)
full_text = "".join(text_out)
print("full text:", full_text)
```

### Embedding example

```python
from llm_sdk.sync_sdk import SyncSDK
from llm_sdk.retries import RetryPolicy
from llm_sdk.providers.sync_registry import ProviderSpec
from llm_sdk_provider_gemini import SyncGeminiClient
from llm_sdk.domain.chat import ChatMessage, ChatPart

sdk = SyncSDK.default()
sdk.registry.register(ProviderSpec(
    name="gemini",
    factory=lambda: SyncGeminiClient(
        location=sdk.settings.gemini.location,
    ),
    models={
        "gemini-2.5-flash",
        "text-multilingual-embedding-002"
    },
))

result = sdk.embed(
    provider="gemini",
    model="text-multilingual-embedding-002",
    input=["Hola mundo", "Hello world"],
)
```

### Async quickstart

```python
import asyncio
from llm_sdk.async_sdk import AsyncSDK
from llm_sdk.domain.chat import ChatMessage, ChatPart

async def main() -> None:
    sdk = AsyncSDK.default()

    # Load installed plugins (entry points)
    sdk.registry.load_plugins()

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


if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming example

```python
import asyncio
from llm_sdk.async_sdk import AsyncSDK
from llm_sdk.domain.chat import ChatMessage, ChatPart


async def main() -> None:
    sdk = AsyncSDK.default()
    sdk.registry.load_plugins()

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


if __name__ == "__main__":
    asyncio.run(main())
```

### Embeddings example

```python
import asyncio
from llm_sdk.async_sdk import AsyncSDK


async def main() -> None:
    sdk = AsyncSDK.default()
    sdk.registry.load_plugins()

    resp = await sdk.embed(
        provider="gemini",
        model="text-multilingual-embedding-002",
        input=["Hola mundo", "Hello world"],
    )

    print(resp.vectors[0][:10])


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Configuration
This provider uses Google credentials from the environment.

### Authentication (Vertex AI)

You must have:
- Google Cloud project enabled for Vertex AI
- Proper credentials available through ADC (Application Default Credentials)

Typical local setup:

```bash
gcloud auth application-default login
```

### Provider-specific settings

This package typically reads:
- Gemini location (example: ```us-central1```)

Depending on your llm-sdk settings design, you may configure:

```bash
export LLM_SDK_LOCATION="us-central1"
```
Or if you implement namespaced settings:
```bash

export LLM_SDK_GEMINI_LOCATION="us-central1"
```

## Project Structure

```
llm-sdk-provider-gemini/
├─ src/
│  └─ llm_sdk_provider_gemini/
│     ├─ plugin.py
│     ├─ settings.py
│     ├─ async_client.py
│     ├─ sync_client.py
│     └─ __init__.py
├─ tests/
│  ├─ test_factory.py
│  └─ test_contracts.py
├─ pyproject.toml
└─ README.md
```

---

## Setup

### Development

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install -U pip
pip install -e .
```

### Install core SDK in editable mode (local dev)
```bash
pip install -e ../llm-sdk
```

### Testing
```bash
pytest -q
```

### Notes

Provider tests should include:
- Factory loads correctly
- Spec metadata is correct
- Client implements required methods
- Error mapping works as expected

Avoid real network tests in unit tests.
If you want integration tests, put them behind an env flag.

---

### Deployment

Build package
```bash
python -m build
```

Upload to TestPyPI
```bash
python -m twine upload --repository testpypi dist/*
```

Upload to PyPI
```bash
python -m twine upload dist/*
```

---

## Roadmap

- Better usage extraction for Gemini responses
- Support for Gemini tool calling (if SDK adds tool abstractions)
- Optional support for non-Vertex Gemini API mode
- Optional response caching hooks

---

## Contributing

PRs welcome.

Recommended workflow:

1. Fork repo
2. Create a feature branch
3. Add tests
4. Run pytest
5. Submit PR

---

## License

MIT License

---

### Contact

Author: Esteban Flores