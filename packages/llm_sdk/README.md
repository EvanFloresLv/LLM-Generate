# llm-sdk

Provider-agnostic Python SDK for Large Language Models (LLMs) with a production-ready async architecture, strong typing, unified errors, retries, logging, and a plugin system based on Python entry points.

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

`llm-sdk` is the **core package** that provides:

- A unified API for multiple LLM providers
- Async-first design (`AsyncSDK`)
- A provider registry that loads providers via **plugins**
- Domain models (requests/responses) independent of providers
- Unified error model
- Retries, timeouts, and structured logging

This package intentionally **does not** ship provider implementations directly.
Providers are installed as separate packages, for example:

- `llm-sdk-provider-gemini`
- `llm-sdk-provider-openai` (future)
- `llm-sdk-provider-ollama` (future)

---

## Architecture

### Core design principles

- **Provider-agnostic domain layer** (no vendor types leak into the SDK)
- **Open/Closed Principle**: providers are added via entry points, without modifying core code
- **SOLID**:
  - `AsyncSDK` orchestrates only
  - Providers implement only provider calls
  - Registry handles discovery
  - Domain models validate input/output
- **DRY/KISS**: minimal abstractions, strong conventions

### Key components

- **AsyncSDK**: main entrypoint (`sdk.chat`, `sdk.embed`, `sdk.stream_chat`)
- **ProviderRegistry**: discovers provider plugins via entry points
- **ProviderFactory**: plugin contract (`spec()` + `create(settings)`)
- **Domain models**: `ChatRequest`, `ChatResponse`, `EmbeddingRequest`, etc.
- **Unified errors**: `ProviderError`, `ValidationError`, `TimeoutError`
- **Retries**: async retry policy with exponential backoff + jitter
- **Settings**: `SDKSettings` (Pydantic Settings)

---

## Features

- Async-first API
- Provider plugins via entry points
- Chat completions
- Embeddings
- Streaming chat (provider-dependent)
- Strong typing (mypy-friendly)
- Unified error handling
- Retry policies with jitter
- Structured logging support
- Configuration via environment variables
- Provider + model resolution at runtime

---

## Installation

### Install core SDK

```bash
pip install llm-sdk
```

Install at least one provider plugin

### Example (Gemini):

```bash
pip install llm-sdk-provider-gemini
```

---

## Usage

### Async Quickstart

```python
import asyncio
from llm_sdk.async_sdk import AsyncSDK


async def main() -> None:
    sdk = AsyncSDK.default()

    # IMPORTANT: load provider plugins (entry points)
    sdk.registry.load_plugins()

    resp = await sdk.chat(
        provider="gemini",
        model="gemini-2.5-flash",
        messages=[("user", "Hola, ¿cómo estás?")],
    )

    print(resp.content)


if __name__ == "__main__":
    asyncio.run(main())
```


### Streaming chat

```python
import asyncio
from llm_sdk.async_sdk import AsyncSDK


async def main() -> None:
    sdk = AsyncSDK.default()
    sdk.registry.load_plugins()

    async for ev in sdk.stream_chat(
        provider="gemini",
        model="gemini-2.5-flash",
        messages=[("user", "Explica qué es un Transformer en 5 puntos.")],
    ):
        print(ev.delta, end="", flush=True)

    print("\n[done]")


if __name__ == "__main__":
    asyncio.run(main())
```

### Embeddings

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

    print(len(resp.vectors), "vectors")
    print("dim:", len(resp.vectors[0]))


if __name__ == "__main__":
    asyncio.run(main())
```

### Listing installed providers (debug)

```python
from llm_sdk.providers.async_registry import ProviderRegistry

reg = ProviderRegistry()
reg.load_plugins()

print(reg.available())
```

---

## Configuration

llm-sdk uses Pydantic Settings.

Example environment variables

```bash
export LLM_SDK_DEFAULT_PROVIDER="gemini"
export LLM_SDK_DEFAULT_MODEL="gemini-2.5-flash"
```

### Typical config options

- Default provider
- Default model
- Retry policy (max attempts, delays)
- Timeouts
- Provider-specific settings (usually defined in provider plugin packages)

---

### Project Structure

Typical structure:

```bash
llm-sdk/
├─ src/
│  └─ llm_sdk/
│     ├─ async_sdk.py
│     ├─ sync_sdk.py
│     ├─ settings.py
│     ├─ typing.py
│     ├─ timeouts.py
│     ├─ exceptions.py
│     ├─ retries.py
│     ├─ plugin_loader.py
│     ├─ lifecycle.py
│     ├─ context.py
│     ├─ domain/
│     │  ├─ chat.py
│     │  ├─ embeddings.py
│     │  └─ models.py
│     └─ providers/
│        ├─ async_base.py
│        ├─ sync_base.py
│        ├─ async_registry.py
│        ├─ sync_registry.py
│        └─ __main__.py
├─ tests/
│  ├─ test_registry.py
│  ├─ test_retries.py
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

pip install -U pip
pip install -e ".[dev]"
```

### Install a provider plugin in editable mode

Example:

```bash
pip install -e ../llm-sdk-provider-gemini
```

### Testing

```bash
pytest -q
```

### Notes
- Core tests should not require any provider network access
- Provider plugins are tested independently in their own packages

---

## Deployment

### Build package

```bash
python -m build
```

### Upload to TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

### Upload to PyPI

```bash
python -m twine upload dist/*
```

---

## Roadmap

- Sync SDK wrapper (LLM) built on top of AsyncSDK
- More providers (OpenAI, Ollama, Anthropic, etc.)
- Built-in observability hooks (OpenTelemetry)
- Tool calling abstraction
- Response caching
- Rate limiting middleware

- First-class tracing

---

## Contributing

PRs are welcome.

### Recommended workflow:
1. Fork the repo
2. Create a feature branch
3. Add tests for changes
4. Run pytest
5. Submit PR

---

## License
MIT License

---

## Contact
Author: Esteban Flores