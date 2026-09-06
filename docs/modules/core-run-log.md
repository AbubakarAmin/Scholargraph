# `core/run_log.py`

## Responsibility

Tracks live run status, append-only events, scratchpad records, and cross-run lessons. It also bridges compatibility JSONL files with the SQLite research ledger.

## Key API

- `RunTracker`: phase, status, stats, messages, completion, and dashboard payload.
- `start_run()` / `get_tracker()`: active process run lifecycle.
- `emit_event()`, `read_events()`, `read_scratchpad()`.
- `CrossRunMemory`: rejection, pivot, revision, and summary lessons.
