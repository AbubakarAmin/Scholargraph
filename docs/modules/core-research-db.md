# `core/research_db.py`

## Responsibility

SQLite persistence for runs, events, scratchpad rows, claims, and generated artifacts.

## Design role

The research ledger is the authoritative durable record for observability and evidence. JSONL files remain compatibility exports and fallback readers.

## Operational note

The default deployment is local and single-process. Multi-user deployment needs a server database, concurrency policy, and authentication.
