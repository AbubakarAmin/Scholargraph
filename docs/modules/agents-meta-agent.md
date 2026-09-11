# `agents/meta_agent.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Evaluates overall run progress and recommends reset, continuation, or stop based on quality, failure signals, and iteration limits.

## Integration

`main.meta_evaluation_node()` supplies the full workflow state and uses the agent decision to route back through narrative writing, reset, or completion.
