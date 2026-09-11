# Testing
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Memory retrieval integrity

```powershell
python -m pytest tests/test_memory_integrity.py -v
```

Adversarial coverage with printed transcripts: fallback/hallucinated-claim poisoning, debate-tag-only retrieval, outcome filtering, provenance, Elo shrinkage, exploration distribution, and a static bypass audit that agents do not call raw memory APIs for prompts.

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

## Reproducibility and incident reports

Replay a generated companion repository in the locked local sandbox:

```powershell
python replay_run.py output/companion_repo
```

Replay in a newly created clean virtual environment:

```powershell
python replay_run.py output/companion_repo --clean-env
```

Generate a report for one run or for the latest failed/completed pair:

```powershell
python forensic_report.py RUN_ID output/forensic_report.json
python historical_report.py output/historical_report.json
```

These reports include events, claims, artifacts, unresolved claims, and claim-to-artifact lineage. They do not bypass source licensing or network policy.

This includes the maintained offline suite and the repository-level compatibility scripts.

## Full legacy checks

```powershell
python test_system.py
```

This is a print-oriented compatibility check. It imports all agents and verifies basic dependencies and directories.

## Live checks

Live runs require a configured `.env`, provider credentials, network access to scholarly APIs, and can consume API quota. Use `python main.py` only after the offline suite passes.
