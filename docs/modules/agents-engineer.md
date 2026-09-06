# `agents/engineer.py`

## Responsibility

Generates experiment code, validates it, executes multi-seed runs, performs cheap branch probes, captures raw artifacts, and chooses REFINE, PIVOT, or plan revision paths.

## Main API

- `run_experiment()` runs the recovery loop.
- `run_branching_search()` probes variants and promotes a winner.
- `request_plan_revision()` and `consume_plan_revision_requests()` form the reverse planning edge.
- `check_code_claim_consistency()` compares prose claims with generated code.

## Safety

All generated code must pass `core.sandbox.validate_code` before execution.
