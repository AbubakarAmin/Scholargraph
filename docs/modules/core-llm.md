## Module Overview

Multi-provider LLM client: Gemini, OpenAI, and any OpenAI-compatible endpoint.
Agents should use call_llm / generate_embedding / get_llm_client — not provider SDKs directly.

# `core/llm.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Provides a provider-neutral LLM and embedding boundary for Gemini, OpenAI, and OpenAI-compatible endpoints.

## Key API

- `LLMClient.chat()`: rate-limited text generation.
- `LLMClient.embed()`: provider embedding call with 768-dimension compatibility.
- `get_llm_client()`: cached client for the active provider.
- `reset_llm_client()`: clears the cache after runtime settings change.

Agents should use this module or the compatibility wrappers in `core.utils`, never provider SDKs directly.
