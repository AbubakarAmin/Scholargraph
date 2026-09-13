# ScholarGraph Documentation

This folder documents the current codebase module by module. The project is a local, multi-agent research workflow that discovers topics, debates hypotheses, plans experiments, validates datasets, executes sandboxed code, independently verifies results, and assembles a LaTeX paper with reproducibility artifacts.

## Start here

- [Architecture](architecture.md): runtime boundaries, data flow, and module map.
- [Workflow](workflow.md): LangGraph phases, state transitions, and QA-mode graph.
- [Configuration](modules/core-config.md): environment and runtime settings.
- [Testing](testing.md): offline checks, validation commands, and test inventory.
- [Improvement status](improvements.md#implementation-status): implementation status and operational limits.

## Module guides

### Entry points

- [main.py](modules/main.md)
- [demo.py](modules/demo.md)
- [run_ui.py](modules/run-ui.md)
- [run_with_real_api.py](modules/run-with-real-api.md)
- [setup.py](modules/setup.md)
- [replay_run.py](modules/replay_run.md)
- [forensic_report.py](modules/forensic_report.md)
- [historical_report.py](modules/historical_report.md)

### Agents

- [Topic Hunter](modules/agents-topic-hunter.md) — citation-graph gap analysis, novelty filter, TopicHunter v2 features
- [Hypothesis Debate](modules/agents-hypothesis-debate.md) — multi-round adversarial debate with ensemble judging and Elo
- [Planner](modules/agents-planner.md) — falsifiable experiment plans with baselines and variants
- [Writer](modules/agents-writer.md) — two-pass narrative and results drafting
- [Engineer](modules/agents-engineer.md) — code generation, sandbox execution, PIVOT/REFINE recovery
- [Data](modules/agents-data.md) — dataset validation, schema checks, provenance hashing
- [Execution](modules/agents-execution.md) — independent seeded replay worker
- [Analysis](modules/agents-analysis.md) — independent SciPy statistical analysis
- [Verification](modules/agents-verification.md) — artifact integrity and statistical mismatch detection
- [Supervisor](modules/agents-supervisor.md) — hard checks + soft reviewer scoring
- [Meta Agent](modules/agents-meta-agent.md) — workflow evaluation and reset policy
- [Editor](modules/agents-editor.md) — LaTeX assembly, bibliography resolution, companion repo

### Core services

- [State](modules/core-state.md) — `ResearchState` TypedDict and initializer
- [Contracts](modules/core-contracts.md) — typed artifact/evidence handoff contracts
- [Context](modules/core-context.md) — `RunContext` and per-run dependency injection
- [Ports](modules/core-ports.md) — persistence port protocols
- [Workflow](modules/core-workflow.md) — LangGraph graph assembly (research + QA)
- [Workflow nodes](modules/core-workflow-nodes.md) — phase node implementations
- [Pipeline](modules/core-pipeline.md) — shared CLI/web execution service
- [Config](modules/core-config.md) — settings, runtime key apply, env sync
- [LLM](modules/core-llm.md) — multi-provider client with cost-aware routing
- [Utils](modules/core-utils.md) — shared helpers, JSON parsing, math validation
- [Memory](modules/core-memory.md) — FAISS vector index and debate/feedback logs
- [Run Log](modules/core-run-log.md) — events, scratchpad, cross-run memory
- [Research DB](modules/core-research-db.md) — SQLite ledger for runs, claims, artifacts
- [Sandbox](modules/core-sandbox.md) — restricted code execution with AST validation
- [Verification](modules/core-verification.md) — citation resolution, statistics, cross-section consistency
- [Sources](modules/core-sources.md) — allowlisted cached scholarly retrieval
- [Evidence Gate](modules/core-evidence_gate.md) — immutable experiment contracts and fail-closed handoffs
- [Evidence Synthesis](modules/core-evidence_synthesis.md) — cross-paper evidence map and bridge validation
- [Capabilities](modules/core-capabilities.md) — role manifests and sandbox capability manifest
- [Tool Broker](modules/core-tool_broker.md) — auditable registered-tool dispatch
- [Datasets](modules/core-datasets.md) — curated local dataset catalog
- [Known Answers](modules/core-known_answers.md) — known-answer validation fixtures
- [Structural Gaps](modules/core-structural_gaps.md) — bibliographic coupling gap analysis
- [Sparsity Matrix](modules/core-sparsity_matrix.md) — method × domain sparsity detection
- [Contradiction Mining](modules/core-contradiction_mining.md) — opposing-claim detection across papers
- [Forensics](modules/core-forensics.md) — durable incident reports
- [Replay](modules/core-replay.md) — clean-environment replay

### Web and tests

- [Web API](modules/web-app.md) — FastAPI Control Deck backend
- [Static UI](modules/web-static.md) — operations console frontend
- [Test suite](testing.md) — offline checks and test inventory
