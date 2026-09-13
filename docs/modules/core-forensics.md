# Forensics — `core/forensics.py`

Durable incident reports for run diagnosis.

## Purpose

Generates structured reports that reconstruct the full pipeline trace for a given run: topic → debate result → plan → committed experiment contract → engineer execution log(s) → verification/supervisor scores → final assembled sections.

## Entry points

### `forensic_report.py`
```bash
python forensic_report.py RUN_ID output/forensic_report.json
```
Generates a per-run report including:
- Run metadata (status, phase, timestamps)
- Event trace from `research_ledger.sqlite`
- Evidence claims and their verification status
- Artifact records and their locations
- Unresolved claims
- Claim-to-artifact lineage

### `historical_report.py`
```bash
python historical_report.py output/historical_report.json
```
Reconstructs the latest failed/completed run pair, including durable claims, artifacts, events, unresolved claims, and lineage.

## Report contents

| Section | Contents |
|---|---|
| `run` | run_id, status, phase, started_at, ended_at |
| `events` | timestamped event trace from the ledger |
| `claims` | evidence claims with type, status, and evidence bundle |
| `artifacts` | artifact records with type, location, and metadata |
| `unresolved` | claims that still need human attention |
| `lineage` | claim-to-artifact mappings for diagnosis |

## Limitations

These reports do not bypass source licensing or network policy. They reconstruct from stored artifacts only.
