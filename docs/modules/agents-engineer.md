## Module Overview

EngineerAgent — verifiable, self-healing experiment runner.
Sandbox lockdown, multi-seed, PIVOT/REFINE, ablations, code-claim checks.

# `agents/engineer.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Generates experiment code, validates it, executes multi-seed runs, performs cheap branch probes, captures raw artifacts, and chooses REFINE, PIVOT, or plan revision paths.

## Main API

- `run_experiment()` runs the recovery loop.
- `run_branching_search()` probes variants and promotes a winner.
- `request_plan_revision()` and `consume_plan_revision_requests()` form the reverse planning edge.
- `check_code_claim_consistency()` compares generated code against experiment contract baselines/claimed_components (not free-text method prose).

## Safety

All generated code must pass `core.sandbox.validate_code` before execution.

## v4.1 upgrades (2026-09)

- **Failure-gradient hints** (`_error_category_hint`): Deterministic hints per failure category (timeout → shrink workload; import → allowed imports; sandbox → no file/subprocess; JSON → metrics line) injected into `_refine_code` prompts.
- **Cross-run lessons**: `_generate_experiment_code` uses `CrossRunMemory().get_prompt_context()` to list prior failure patterns. Never uses `lessons_for_prompt()` — the memory-integrity static audit forbids it.
