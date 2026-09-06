# Architecture

## Purpose

ScholarGraph is a local research-paper generation system. It combines LLM-assisted planning and writing with deterministic checks and executable experiments.

## Runtime layers

1. **Entry points** start either the CLI, demo, real-API wrapper, or FastAPI UI.
2. **Orchestration** in `main.py` builds a LangGraph state machine.
3. **Agents** perform domain work: discovery, debate, planning, writing, engineering, review, meta-evaluation, and editing.
4. **Core services** provide configuration, LLM access, memory, logging, sandbox execution, persistence, and verification.
5. **Artifacts** are written to `output/` and durable run data is stored in SQLite and JSONL compatibility logs.

## Important boundaries

- Agents call `core.llm` for provider access and receive the active `RunContext`; provider SDKs should not spread into agent modules.
- `core.state.ResearchState` is the shared workflow contract.
- `core.contracts` defines typed handoffs and `core.context.RunContext` carries per-run dependencies.
- `core.workflow` owns graph assembly and `core.pipeline` owns graph execution for CLI and web.
- `core.verification` contains deterministic checks. LLM review is advisory and cannot rescue a hard failure.
- `core.sandbox` is a local lockdown mechanism, not a security boundary.
- `core.research_db` is the durable source of truth for runs, events, claims, and artifacts.
- `web.app` reads workflow state and persistence services but starts the pipeline in a background thread.

## Main data flow

`TopicHunterAgent -> HypothesisDebateSystem -> PlannerAgent -> WriterAgent (narrative) -> EngineerAgent -> WriterAgent (results) -> SupervisorAgent -> MetaAgent or EditorAgent`

State is passed between these phases as a mutable `ResearchState` dictionary. An engineering failure can send revision requests back to the planner. Results writing can redraft when numeric grounding fails.

## Current coupling risks

- `main.py` still contains graph construction, node implementations, CLI presentation, checkpoint creation, and result serialization.
- Several agents write directly to global stores and read global configuration.
- The web server and CLI each own a similar pipeline execution loop.
- Some older agent code uses compatibility helpers from `core.utils` instead of the provider-neutral `core.llm` API.

These are recorded as staged work in [the refactor roadmap](refactor-roadmap.md).
