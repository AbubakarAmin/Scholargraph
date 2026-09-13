# Evidence Gate — `core/evidence_gate.py`

Immutable experiment contracts and fail-closed handoffs.

## Purpose

The evidence gate ensures that once an experiment contract is committed, no downstream agent can silently change the dataset, requirements, metrics, baselines, or hypothesis. Any such change is contract drift and requires a new experiment identity.

## Key functions

### `validate_experiments(experiments)`

Deterministic validation of a plan's experiment list:
- Each experiment must be a dict with a non-empty `name`
- Names must be unique
- Each experiment must declare `evaluation_metrics`
- Variants must be dicts

Returns a list of error strings (empty = valid).

### `build_contract(experiment, dataset)`

Creates an immutable `ExperimentContract` from an `ExperimentSpec` and optional dataset. The contract includes:
- `experiment_name`
- `hypothesis` (from `falsifiable_prediction` or `description`)
- `dataset` snapshot
- `requirements` (type, methodology, claimed_components)
- `baselines`
- `evaluation_metrics`
- `split_policy`
- `seeds`
- `stopping_rule`
- `analysis_protocol` (primary_metric, statistical_test, effect_size, confidence_level)
- `contract_hash`: SHA-256 of the deterministic contract content

### `gate_engineering_outputs(state)`

Post-engineering gate that checks:
- All committed contracts have matching outputs
- No contract hash has been modified
- Raw result files exist
- Returns an `EvidenceGateDecision` with `allowed`, `terminal`, `reason_code`, and `message`

### `validate_dataset_identity(contract, artifact)`

Ensures the dataset used during execution matches the committed contract (by path and hash).

### `validate_experiments(experiments, contracts)`

Verifies that executed experiments match their committed contracts.

## Contract drift

If any downstream agent attempts to change a committed contract field, the gate rejects the change. The only valid path is to request a plan revision through `EngineerAgent.request_plan_revision()`, which bounces back to the planning phase with a new contract.
