# `core/state.py`

## Responsibility

Defines the typed state contract shared by every LangGraph node and creates the initial state for a new run.

## Key API

- `ResearchState`: TypedDict containing phase control, topics, debate results, plan, drafts, experiment outputs, quality results, and final artifacts.
- `initialize_state()`: returns a fresh state with empty collections and safe defaults.

## Design rule

Changes to workflow state should start here. Keep node code dependent on this contract rather than duplicating fields in orchestration modules.
