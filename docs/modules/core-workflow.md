# `core/workflow.py`

## Responsibility

Builds the LangGraph state machine and owns phase routing. Node implementations are injected as a mapping so the composition root can migrate them independently.

## Key API

`create_research_graph(nodes)` returns an uncompiled `StateGraph`. The mapping contains phase nodes plus `should_reset` and `should_continue` routing functions.

## Compatibility

`main.create_research_graph()` remains the public wrapper used by CLI, web, demo, and tests. It delegates graph assembly to this module.
