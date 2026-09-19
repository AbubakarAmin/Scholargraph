## Module Overview

Composition root for the multi-agent research system.

# `main.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Composition root for the CLI research workflow. It defines the graph nodes, graph routing, checkpoint setup, output serialization, and command-line progress display.

## Public surface

- `ResearchState` and `initialize_state` are compatibility re-exports from `core.state`.
- `create_research_graph()` builds and returns the LangGraph state graph.
- `create_checkpointer()` opens the configured SQLite checkpoint store.
- `save_results()` writes LaTeX, plan, and summary artifacts.
- `main()` validates configuration and streams a CLI run.

## Refactor note

Graph construction and node wrappers live in `core/workflow.py` and `core/workflow_nodes.py`. Shared CLI/web execution lives in `core/pipeline.py`. This file retains the CLI entry point and compatibility re-exports.
