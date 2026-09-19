## Module Overview

Configuration module for the multi-agent research system.
Handles environment variables, API keys, and system settings.

# `core/config.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Loads environment settings, exposes the process-wide `config` object, applies UI runtime settings, and synchronizes selected values into `.env`.

## Key API

- `Config`: Pydantic settings model and model-tier resolution.
- `config`: global runtime configuration.
- `validate_config()`: checks provider and required source settings.
- `apply_runtime_keys()`: updates environment and live config values.
- `sync_env_file()`: upserts UI-managed settings.

## Key settings

| Setting | Default | Purpose |
|---|---|---|
| `llm_provider` | `gemini` | `gemini` \| `openai` \| `openai_compatible` |
| `sandbox_backend` | `docker` | `ast` (in-process) or `docker` execution |
| `supervisor_threshold` | `8.5` | Mean section score → editing |
| `debate_pass_threshold` | `7.0` | Debate PASS floor |
| `seed_sequential` | `true` | Process seeds sequentially |
| `persona_sequential` | `true` | Process personas sequentially |
| `llm_budget_per_seed` | `30` | Max LLM calls per seed thread |
| `novelty_max_abstracts` | `20` | Max abstracts for embedding precomputation |

## Gotchas

- `.env` is loaded from project root (not cwd) via `dotenv` at line 14.
- Config defaults in code may differ from `env_example.txt` (e.g., `DEBATE_PASS_THRESHOLD` is `7.0` in code, `7.5` in env_example.txt). Code wins at runtime.
- Matplotlib is forced to `Agg` backend process-wide (line 18) to avoid Tcl crashes on Windows.

## Operational warning

`memory/keys.json` stores local secrets. Keep it out of version control and do not expose the web server outside a trusted machine without authentication.
