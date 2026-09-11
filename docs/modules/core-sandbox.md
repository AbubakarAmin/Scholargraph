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

## Security boundary

This is a local safety mechanism for generated experiments, not a hardened isolation boundary. Do not treat it as equivalent to a container or remote sandbox.
