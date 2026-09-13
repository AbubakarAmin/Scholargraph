# Testing

## Focused offline suite

```bash
python -m pytest tests/test_eval_harness.py -q
```

This is the primary measurement surface for "did this upgrade help?" It covers sandbox restrictions, multi-seed aggregation, citation extraction, statistical verification, planner gates, code-claim consistency, cross-run memory, configuration, debate shape, source outage handling, SQLite persistence, reproducibility checks, and more.

## Full pytest collection

```bash
python -m pytest -q
```

The full suite covers (all offline/mocked where needed):

| Test file | Coverage |
|---|---|
| `test_eval_harness.py` | Sandbox, multi-seed, citations, stats, planner, code-claim, cross-run memory, config, debate, source outage, SQLite, reproducibility |
| `test_capabilities.py` | Capability manifests, authorization, sandbox capability manifest |
| `test_sources.py` | SourceClient allowlisting, caching, retries, outages |
| `test_data_agent.py` | DatasetAgent hashing, schema, target checks, formats |
| `test_execution_agent.py` | ExecutionAgent seeded replay, artifact creation, forbidden code |
| `test_analysis_agent.py` | AnalysisAgent CIs, Welch tests, effect sizes, warnings |
| `test_verification_agent.py` | VerificationAgent hash, path, statistical mismatch |
| `test_refactor_boundaries.py` | Module boundary enforcement |
| `test_evidence_gate.py` | Contract building, validation, dataset identity |
| `test_evidence_synthesis.py` | Cross-paper evidence maps, bridge validation |
| `test_qa_mode.py` | QA literature retrieval, answer generation |
| `test_topic_hunter_v2.py` | TopicHunter v2 core features |
| `test_topic_hunter_features_8_15.py` | Persona ensemble, seed-strategy Elo, frontier seeding |
| `test_memory_integrity.py` | Adversarial memory retrieval, poisoning, bypass audit |
| `test_power_and_rescope.py` | Prospective power preregistration, dataset rescoping |
| `test_container_sandbox.py` | Docker sandbox backend compatibility |
| `test_sandbox_allowlist.py` | Sandbox import allowlist enforcement |
| `test_datasets_openreview.py` | OpenReview calibration dataset loading |
| `test_planner_manifest.py` | Planner feasibility checks against capability manifest |
| `test_run_review_fixes.py` | Run review hardening fixes |
| `test_run_review_fixes_2.py` | Additional run review fixes |
| `test_extract_local_code_structure.py` | Traceback source extraction |
| `test_metrics_parsing.py` | Metric extraction from stdout |
| `test_arxiv_sleep_guard.py` | arXiv rate-limit backoff |
| `test_remaining_implementation.py` | Remaining implementation coverage |
| `tests/smoke_offline.py` | Quick standalone smoke check (no pytest) |

## Smoke script

```bash
python tests/smoke_offline.py
```

The smoke script exercises the same core paths without requiring live provider keys.

## Reproducibility and incident reports

Replay a generated companion repository in the locked local sandbox:

```bash
python replay_run.py output/companion_repo
```

Replay in a newly created clean virtual environment:

```bash
python replay_run.py output/companion_repo --clean-env
```

Generate a report for one run or for the latest failed/completed pair:

```bash
python forensic_report.py RUN_ID output/forensic_report.json
python historical_report.py output/historical_report.json
```

These reports include events, claims, artifacts, unresolved claims, and claim-to-artifact lineage. They do not bypass source licensing or network policy.

## Full legacy checks

```bash
python test_system.py
```

This is a print-oriented compatibility check. It imports all agents and verifies basic dependencies and directories.

## Live checks

Live runs require a configured `.env`, provider credentials, network access to scholarly APIs, and can consume API quota. Use `python main.py` only after the offline suite passes.
