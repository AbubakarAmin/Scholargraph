## Module Overview

Restricted Python sandbox for Engineer experiments.
Blocks exit(), subprocess, os.system, and other dangerous calls.
Does NOT rely on the model behaving — AST + builtins lockdown.

# `core/sandbox.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Validates and executes generated experiment code under an AST and builtins lockdown, then aggregates metrics across seeds.

## Key API

- `validate_code()`: rejects dangerous names, modules, and calls.
- `execute_sandboxed()`: runs one experiment with captured output.
- `run_multi_seed()`: repeats execution and calculates aggregate metrics.

## Backends

The sandbox supports two execution backends, selected via `SANDBOX_BACKEND`:

| Backend | Module | Isolation | Use case |
|---|---|---|---|
| `ast` (default fallback) | `core/sandbox.py` | In-process AST + restricted builtins | No Docker dependency; lightweight |
| `docker` | `core/container_sandbox.py` | Docker container (`--network none`, `--read-only`) | Stronger isolation; resource limits |

Selection is handled by `core/sandbox_dispatch.py`, which automatically falls back to `ast` if Docker is unreachable.

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| `SANDBOX_BACKEND` | `"docker"` | `"ast"` or `"docker"` |
| `SANDBOX_TIMEOUT_SEC` | `120` | Soft timeout budget |
| `SANDBOX_MAX_OUTPUT_BYTES` | `1048576` | Output truncation limit |
| `SANDBOX_DOCKER_MEMORY` | `"2g"` | Docker memory limit |
| `SANDBOX_DOCKER_CPUS` | `"1.0"` | Docker CPU limit |

## Security boundary

This is a local safety mechanism for generated experiments, not a hardened isolation boundary. Do not treat it as equivalent to a container or remote sandbox.
