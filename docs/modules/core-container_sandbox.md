# Container Sandbox

**File:** `core/container_sandbox.py` (313 lines)

## Purpose

Docker-based execution backend for Engineer experiments, providing stronger isolation than the AST sandbox: no network, read-only filesystem, configurable memory/CPU limits.

## Key functions

| Function | Purpose |
|---|---|
| `execute_containerized(code, timeout_sec, seed, memory, cpus)` | Runs code in a Docker container with `--network none`, `--read-only`, `--tmpfs /tmp`, resource limits |
| `container_multi_seed(code, n_seeds, base_seed, ...)` | Runs N seeds sequentially in containers and aggregates metrics (mean/std) |
| `_build_preamble(seed)` | Generates imports and seed initialization code prepended to every experiment |
| `_parse_json_from_stdout(stdout)` | Extracts the last JSON object from stdout (scans backwards for balanced braces) |

## Container configuration

- Image: `scholargraph-sandbox:latest` (must be pre-built)
- Flags: `--network none` (no network access), `--read-only` filesystem, `--tmpfs /tmp`
- Resource limits: configurable via `SANDBOX_DOCKER_MEMORY` (default `"2g"`) and `SANDBOX_DOCKER_CPUS` (default `"1.0"`)
- Output truncation respects `config.sandbox_max_output_bytes`

## Gotchas

- On timeout, kills the `docker run` process and force-removes any leftover containers from the image.
- Multi-seed aggregation returns a combined error message listing per-seed failure details if all seeds fail.
- Falls back to AST sandbox automatically if Docker is unreachable (via `sandbox_dispatch.py`).
- This is a process-level sandbox, not a security boundary for hostile code.
