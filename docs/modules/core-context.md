# `core/context.py`

## Responsibility

Defines `RunContext`, the dependency bundle for one research run: configuration, vector memory, research ledger, tracker, and run identity.

## Key API

`create_run_context(tracker)` creates a context from the current application adapters. `ResearchPipeline` stores the context and updates its run ID when streaming.

## Migration role

All primary agents and the Elo persistence helper accept this object. Existing globals remain default adapters only when a component is constructed outside a pipeline run.
