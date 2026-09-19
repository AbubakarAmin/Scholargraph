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
- `call_llm(prompt, tier=..., model=..., temperature=...)`: Primary entry; agents should use this.
- `generate_embedding(text)`: Gemini embed or OpenAI embed → 768-d.

## Cost-aware routing

Tiers: `cheap` (lookups/novelty), `strong` (debate/plan), `judge` (ensemble / peer review).

Optional overrides: `LLM_MODEL_CHEAP`, `LLM_MODEL_STRONG`, `LLM_MODEL_JUDGE`.

## v4 upgrade (2026-09)

- **LLM failure visibility**: `llm_failures` run stat + error-level message on chat failure. Visible in run log and dashboard.

## Gotchas

- Rate limit: `LLM_REQUEST_INTERVAL` seconds between calls (default `1.0`).
- `setup_gemini()` / `call_gemini()` are **legacy aliases** → same client (Writer etc. still work).
- Agents should use this module or the compatibility wrappers in `core.utils`, never provider SDKs directly.
