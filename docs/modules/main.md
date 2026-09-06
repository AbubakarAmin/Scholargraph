# `main.py`

## Responsibility

Composition root for the CLI research workflow. It defines the graph nodes, graph routing, checkpoint setup, output serialization, and command-line progress display.

## Public surface

- `ResearchState` and `initialize_state` are compatibility re-exports from `core.state`.
- `create_research_graph()` builds and returns the LangGraph state graph.
- `create_checkpointer()` opens the configured SQLite checkpoint store.
- `save_results()` writes LaTeX, plan, and summary artifacts.
- `main()` validates configuration and streams a CLI run.

## Refactor note

This file is currently the largest orchestration surface. The next safe extraction is `workflow_nodes.py`, followed by a shared pipeline runner for CLI and web.
