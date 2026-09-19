# Sandbox Dispatch

**File:** `core/sandbox_dispatch.py` (110 lines)

## Purpose

Single-point dispatch that routes sandbox execution calls (`execute`, `execute_multi_seed`) to either the AST/builtins sandbox or the Docker container sandbox based on `config.sandbox_backend`.

## Key functions

| Function | Purpose |
|---|---|
| `_docker_daemon_reachable()` | One-time process-level probe of the Docker daemon (cached result) |
| `_use_docker()` | Returns `True` only if config says "docker" AND the daemon is reachable |
| `execute(code, timeout_sec, seed)` | Dispatches to `execute_containerized` (Docker) or `execute_sandboxed` (AST) |
| `execute_multi_seed(code, n_seeds, base_seed)` | Dispatches to `container_multi_seed` or `run_multi_seed` |

## Configuration

- `SANDBOX_BACKEND` env var: `"docker"` (default) or `"ast"`
- If Docker is configured but unreachable, falls back to the AST sandbox with a one-time warning (never crashes)
- Docker probe result is cached for the process lifetime (thread-safe via lock)

## Gotchas

- Set `SANDBOX_BACKEND=ast` explicitly for in-process execution (avoids Docker dependency entirely).
- The dispatch is transparent — callers use `execute()` / `execute_multi_seed()` and get the same return format regardless of backend.
