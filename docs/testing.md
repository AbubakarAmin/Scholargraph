# Testing

## Focused offline suite

```powershell
python -m pytest tests/test_eval_harness.py -q
```

This suite covers sandbox restrictions, multi-seed aggregation, citation extraction, statistical verification, planner gates, code-claim consistency, cross-run memory, configuration, debate shape, source outage handling, SQLite persistence, and reproducibility checks.

## Smoke script

```powershell
python -m tests.smoke_offline
```

The smoke script exercises the same core paths without requiring live provider keys.

## Full pytest collection

```powershell
python -m pytest -q
```

This includes the maintained offline suite and the repository-level compatibility scripts.

## Full legacy checks

```powershell
python test_system.py
```

This is a print-oriented compatibility check. It imports all agents and verifies basic dependencies and directories.

## Live checks

Live runs require a configured `.env`, provider credentials, network access to scholarly APIs, and can consume API quota. Use `python main.py` only after the offline suite passes.
