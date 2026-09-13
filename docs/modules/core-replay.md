# Replay — `core/replay.py`

Clean-environment replay for companion repositories.

## Purpose

Provides the ability to replay a generated companion repository in a clean environment, verifying that the experiment code is self-contained and produces the same results.

## Entry point

`replay_run.py` at the project root:
```bash
python replay_run.py output/companion_repo          # replay in current env
python replay_run.py output/companion_repo --clean-env  # replay in fresh virtualenv
```

## What it does

1. Reads the companion repository's `requirements.txt` and `run_experiments.py`
2. Optionally creates a fresh virtualenv (`--clean-env`)
3. Installs dependencies
4. Executes the experiment scripts in the sandbox
5. Compares output metrics against the original raw results
6. Reports pass/fail with detailed comparison

## Relationship to forensic reports

- `replay_run.py` re-executes code and verifies output
- `forensic_report.py` generates a static incident report from stored artifacts
- `historical_report.py` reconstructs the latest failed/completed run pair
