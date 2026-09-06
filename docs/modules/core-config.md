# `core/config.py`

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
