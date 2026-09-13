# ExecutionAgent — `agents/execution.py`

Independent seeded replay worker.

## Purpose

`ExecutionAgent` wraps `core.sandbox.run_multi_seed` to replay experiment code in a controlled environment. It validates code before execution, captures metadata, writes raw result JSON, and produces an `ExecutionArtifact` with a content hash.

## When it runs

In the `independent_validation` phase, after engineering and before analysis/verification.

## What it does

1. Validates code via `core.sandbox.validate_code` (AST check)
2. Runs the requested seed set via `run_multi_seed`
3. Captures Python version, platform, and sandbox metadata
4. Writes raw result JSON to `output/raw_results/`
5. Produces an `ExecutionArtifact` with:
   - `raw_results_path`: location of raw JSON
   - `content_hash`: SHA-256 of the raw results
   - `environment`: platform and version info
   - `seed_results`: per-seed execution outputs
   - `status`: `success` or `failure`
   - `error`: error message if failed

## What it does NOT do

- Does not generate code (Engineer's responsibility)
- Does not interpret scientific meaning (Analysis's responsibility)
- Does not compute statistics (Analysis's responsibility)
- Does not download datasets (DataAgent's responsibility)

## Capability manifest

```
Allowed: artifact.read, code.execute, artifact.write
Forbidden: code.generate, dataset.download, analysis.statistics, verification.claims
```
