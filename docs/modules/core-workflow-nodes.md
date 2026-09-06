# `core/workflow_nodes.py`

This module contains the phase implementations used by the LangGraph research workflow. Each node receives and mutates a `ResearchState`, uses the active `RunContext` when constructing agents, and returns the updated state.

## Responsibilities

- Discover and rank research topics.
- Debate hypotheses and select a topic.
- Create or revise plans.
- Draft narrative and results sections.
- Run experiments and handle plan revision requests.
- Supervise, evaluate, edit, and reset workflow state.
- Provide route selectors for reset and continuation edges.

Graph construction remains in [`core/workflow.py`](core-workflow.md). The composition root [`main.py`](main.md) injects these nodes into the graph and retains compatibility wrappers for direct callers.
