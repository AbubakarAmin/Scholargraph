# `core/workflow_nodes.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

This module contains the phase implementations used by the LangGraph research workflow. Each node receives and mutates a `ResearchState`, uses the active `RunContext` when constructing agents, and returns the updated state.

## Responsibilities

- Discover and rank research topics.
- Debate hypotheses and select a topic.
- Create or revise plans.
- Draft narrative and results sections.
- Run experiments and handle plan revision requests.
- Supervise, evaluate, edit, and reset workflow state.
- Provide route selectors for reset and continuation edges.
- QA-mode literature retrieval, answer synthesis, and verification.

## v4 node additions (2026-09)

| Node/Function | Purpose |
|---|---|
| `section_revision_feedback` | Carries deterministic check failures + supervisor feedback into writer redrafts |
| `editor_repair_route` | Routes editor release-referee failures to one bounded repair pass |
| `qa_answer_node` | QA-mode synthesis with `call_llm_json` parse-error re-ask |
| `write_narrative_sections` | Narrative revision loop with `narrative_revision_count` bound |
| `write_results_sections` | Results redrafts with feedback-aware revision |
| `editing_node` | Editing with `editor_repair_count` tracking |

Graph construction remains in [`core/workflow.py`](core-workflow.md). The composition root [`main.py`](main.md) injects these nodes into the graph and retains compatibility wrappers for direct callers.
