## Module Overview

Runtime dependencies shared by one research run.

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

## Operational warning

`memory/keys.json` stores local secrets. Keep it out of version control and do not expose the web server outside a trusted machine without authentication.
