## Module Overview

Shared state contract for the ScholarGraph workflow.

# `core/state.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Defines the typed state contract shared by every LangGraph node and creates the initial state for a new run.

## Key API

- `ResearchState`: TypedDict containing phase control, topics, debate results, plan, drafts, experiment outputs, quality results, and final artifacts.
- `initialize_state()`: returns a fresh state with empty collections and safe defaults.

## v4 state fields

| Field | Type | Purpose |
|---|---|---|
| `narrative_revision_count` | `int` | Tracks narrative revision attempts (bound = 1) |
| `editor_repair_count` | `int` | Tracks editor repair attempts (bound = 1) |
| `editor_repair_findings` | `List[str]` | Findings from editor repair pass |

## Design rule

Changes to workflow state should start here. Keep node code dependent on this contract rather than duplicating fields in orchestration modules.
