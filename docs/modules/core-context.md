# Context

**File:** `core/context.py` (59 lines)

## Purpose

Provides a `RunContext` dataclass and a `ContextVar`-based mechanism for sharing runtime dependencies (config, memory, research DB, tracker) across a single pipeline execution.

## Key classes

| Class/Function | Purpose |
|---|---|
| `RunContext` | Dataclass holding `config`, `memory`, `research_db`, `tracker`, `run_id`, and `capability_manifests` for one research run |
| `create_run_context(tracker)` | Factory that wires in default config, memory, research DB, and capability manifests |
| `activate_context(context)` | Sets the `ContextVar` so downstream code can access the current run's dependencies |
| `reset_context(token)` | Restores the previous context (for streaming/graph execution) |
| `get_active_context()` | Retrieves the current `RunContext` or `None` |

## Usage

```python
from core.context import create_run_context, activate_context, reset_context, get_active_context

ctx = create_run_context(tracker)
token = activate_context(ctx)
try:
    # ... pipeline execution ...
    current = get_active_context()
finally:
    reset_context(token)
```

## Gotchas

- Uses `contextvars.ContextVar`, so context is task-local — safe for async but requires explicit `activate_context`/`reset_context` bracketing around each pipeline invocation.
- `RunContext` carries capability manifests so agents can check permissions without reaching into global config.
