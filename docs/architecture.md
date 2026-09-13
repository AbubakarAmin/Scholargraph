# Architecture

## Purpose

ScholarGraph is a local research-workflow system. It combines LLM-assisted planning and writing with deterministic checks, sandboxed experiment execution, independent validation, and evidence-backed verification.

## Runtime layers

1. **Entry points** start either the CLI (`main.py`), demo (`demo.py`), real-API wrapper (`run_with_real_api.py`), or FastAPI UI (`run_ui.py` → `web/app.py`).
2. **Orchestration** in `main.py` builds a LangGraph state machine via `core/workflow.py`. A separate QA-mode graph handles literature synthesis queries.
3. **Agents** perform domain work: discovery, debate, planning, data validation, writing (narrative + results), engineering, independent validation (execution → analysis → verification), supervision, meta-evaluation, and editing.
4. **Core services** provide configuration, LLM access, memory, logging, sandbox execution, persistence, verification, evidence gating, evidence synthesis, dataset catalog, source retrieval, and capability brokering.
5. **Artifacts** are written to `output/` and durable run data is stored in SQLite (`core/research_db.py`) with JSONL compatibility exports.

## Important boundaries

- Agents call `core.llm` for provider access and receive the active `RunContext`; provider SDKs should not spread into agent modules.
- `core.state.ResearchState` is the single shared workflow contract.
- `core.contracts` defines typed handoffs (`ExperimentContract`, `DatasetArtifact`, `ExecutionArtifact`, `StatisticalReport`, `VerificationFinding`, etc.) and `core.context.RunContext` carries per-run dependencies.
- `core.workflow` owns graph assembly and `core.pipeline.ResearchPipeline` owns graph execution for both CLI and web.
- `core.evidence_gate` owns immutable experiment contracts and fail-closed handoffs. LLM review is advisory and cannot rescue a hard failure.
- `core.evidence_synthesis` builds auditable cross-paper evidence maps with cited bridges; candidates must cite valid bridge IDs.
- `core.verification` contains deterministic statistical, provenance, and manuscript checks. LLM review is advisory and cannot rescue a hard failure.
- `core.sandbox` is a local AST/builtins lockdown mechanism, not a security boundary. Supports both `ast` (in-process) and `docker` backends via `SANDBOX_BACKEND`.
- `core.datasets` is the local-only dataset catalog; planners may use catalogued assets or generated synthetic data, never implicit downloads.
- `core.sources` caches allowlisted source responses. Full text requires an explicit open-access/license signal.
- `core.known_answers` provides known-answer validation fixtures for experiment code before trust.
- `core.replay` and `core.forensics` provide clean-environment replay and durable incident reports.
- `core.research_db` is the durable source of truth for runs, events, claims, and artifacts.
- `core.capabilities` defines role manifests and a `SandboxCapabilityManifest` shared by planning, debate, and engineering.
- `core.tool_broker` authorizes registered tool calls, fails closed for unknown capabilities, and records audit entries.
- `core.ports` defines persistence port protocols (`ResearchLedgerPort`, `VectorMemoryPort`) for future adapter injection.
- `core.structural_gaps`, `core.sparsity_matrix`, and `core.contradiction_mining` provide additional gap signals for TopicHunter v2.
- `web.app` reads workflow state and persistence services but starts the pipeline in a background thread.

## Main data flow

```
TopicHunterAgent (with evidence synthesis + structural gap mining)
  → HypothesisDebateSystem (adversarial + ensemble + Elo)
    → PlannerAgent (falsifiable plan + feasibility check)
      → DataAgent (dataset validation + provenance)
        → WriterAgent (narrative sections: Intro, Related Work, Methods)
          → EngineerAgent (code generation + sandbox execution + branch search)
            → Independent Validation:
                ExecutionAgent → AnalysisAgent → VerificationAgent
              → WriterAgent (results sections: Results, Discussion, Abstract)
                → SupervisorAgent (hard checks + soft reviewer)
                  → MetaAgent (continue/reset) or EditorAgent (LaTeX + artifacts)
```

State is passed between these phases as a mutable `ResearchState` dictionary. Before engineering, each experiment receives a content-hashed contract and dataset identity (`core.evidence_gate`). Code repairs may change implementation only; contract drift is terminal. Technical execution failure produces a failure dossier and stops downstream agents. Results writing can redraft numeric grounding, but cannot invent measurements or override the evidence gate.

### QA-mode flow

A separate graph (`create_qa_graph`) runs: Literature Retrieval → Synthesis Answer → Verification. Invoked via `--mode qa --query "..."`. Skips hypothesis debate, planning, engineering, and evidence-gate machinery entirely. Citation grounding is enforced inside the answer node.

## Current coupling risks

- `main.py` still contains graph construction, node wrapper implementations, CLI presentation, checkpoint creation, and result serialization.
- Several agents write directly to global stores and read global configuration.
- The web server and CLI each own a similar pipeline execution loop.
- Some older agent code uses compatibility helpers from `core.utils` instead of the provider-neutral `core.llm` API.

## Release guarantees

The editor runs deterministic citation, numeric, dataset, checklist, reproducibility, and failure checks before assembly. A manuscript is reviewable but not publishable until a human calls `POST /api/release/approve`; approval is persisted in the research ledger and triggers artifact export.

## Module map

| Directory | Key files | Purpose |
|---|---|---|
| `agents/` | `topic_hunter.py`, `hypothesis_debate.py`, `planner.py`, `writer.py`, `engineer.py`, `data.py`, `execution.py`, `analysis.py`, `verification.py`, `supervisor.py`, `meta_agent.py`, `editor.py` | One file per agent role |
| `core/` | `config.py`, `llm.py`, `state.py`, `workflow.py`, `workflow_nodes.py`, `pipeline.py`, `context.py`, `contracts.py`, `verification.py`, `sandbox.py`, `sources.py`, `datasets.py`, `evidence_gate.py`, `evidence_synthesis.py`, `capabilities.py`, `tool_broker.py`, `research_db.py`, `run_log.py`, `memory.py`, `ports.py`, `utils.py`, `known_answers.py`, `structural_gaps.py`, `sparsity_matrix.py`, `contradiction_mining.py`, `forensics.py`, `replay.py` | Shared services |
| `web/` | `app.py`, `static/admin/` | FastAPI backend + operations console |
| `tests/` | 26 test files + `smoke_offline.py` | Offline eval harness and per-module tests |
| `output/` | `raw_results/`, `companion_repo/`, `source_cache/` | Generated artifacts (gitignored) |
| `memory/` | `vector_db/`, `keys.json`, `cross_run.jsonl`, `elo_ratings.json`, `checkpoints.sqlite`, `research_ledger.sqlite` | Durable run state (gitignored) |
