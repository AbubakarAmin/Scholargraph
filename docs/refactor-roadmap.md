# Refactor Roadmap

## Completed in this pass

- Extracted the shared `ResearchState` TypedDict and `initialize_state()` into `core/state.py`.
- Extracted LangGraph graph construction and routing into `core/workflow.py` with injected node callables.
- Added `core/pipeline.py` and routed both CLI and web execution through the shared runner.
- Added typed handoff contracts in `core/contracts.py` and applied them to `ResearchState`.
- Added `RunContext` in `core/context.py` and persistence protocols in `core/ports.py`; CLI and web create one context per run.
- Activated `RunContext` around pipeline streams and injected it into every primary agent, with compatibility support for direct zero-argument test doubles.
- Migrated agent configuration, vector memory, feedback, ledger, and Elo accesses to context-backed dependencies with direct-call fallbacks.
- Extended `ResearchPipeline.run()` to own final-state capture, artifact finalization, and tracker completion for CLI and web.
- Extracted result serialization into `core/artifacts.py` while preserving `main.save_results()` compatibility.
- Migrated `WriterAgent` from legacy Gemini-only helpers to `core.llm`.
- Preserved compatibility imports from `main.py`, so existing callers and tests continue to work.
- Added a documentation index, architecture guide, workflow guide, and one module guide per source surface.
- Extracted all phase node implementations into `core/workflow_nodes.py`; `main.py` is now a thin composition root with compatibility wrappers.
- Added offline boundary regression tests for artifacts, context activation, pipeline lifecycle, graph compilation, and route selectors.
- Completed Stage 5 contract migration at agent boundaries with shared `TypedDict` contracts and regression assertions.
- Consolidated LLM compatibility aliases in `core.llm`, removed their duplicate `core.utils` re-exports, and updated the Gemini integration script.
- Added a filesystem-backed `ArtifactPort` implementation with traversal-safe writes and offline regression coverage.

## Stage status

### Stage 1: orchestration extraction

Completed. Phase node implementations now live in `core/workflow_nodes.py`; `main.py` imports them, builds the graph, and handles CLI concerns.

### Stage 2: runtime services

Completed. Agent internals now use context-backed dependencies; compatibility fallbacks remain only for isolated direct construction.

### Stage 3: pipeline runner

Completed. `core/pipeline.py` now owns graph execution, final-state capture, optional artifact finalization, and tracker completion. `stream()` remains available for live consumers.

### Stage 4: artifact and persistence ports

Completed. Agent ledger/memory writes use context adapters, result serialization has a dedicated service, and `FilesystemArtifactStore` implements `ArtifactPort` with boundary tests.

### Stage 5: agent contracts

Completed for workflow-facing agent boundaries. Planner, Writer, Editor, Engineer, Supervisor, Topic Hunter, and Meta Agent now use shared contracts for their primary inputs and outputs. Broad dictionaries remain only where provider-specific payloads or internal helper structures are intentionally open-ended.

### Stage 6: compatibility cleanup

Completed for production code. LLM aliases now live only in `core.llm`; `core.utils` no longer re-exports provider access, and the integration script uses the current client API. Direct-script `sys.path` setup and `.env` loading remain intentional entry-point bootstrap behavior, not duplicate provider setup.

## Refactor rules

- Make one boundary change at a time.
- Preserve CLI and web behavior after each stage.
- Add an offline test before changing a gate or persistence contract.
- Do not move generated data into source packages.
