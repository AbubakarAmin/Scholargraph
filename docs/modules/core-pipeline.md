# `core/pipeline.py`

## Responsibility

Provides the shared execution service used by CLI and web entry points. It compiles the injected graph, creates the configured checkpoint store, applies the recursion limit, and yields node outputs.

## Key API

`ResearchPipeline(graph_factory, checkpointer_factory).stream(initial_state, run_id, resume=False)` provides live node events. `run(...)` consumes the stream, returns `PipelineResult`, invokes optional finalization, and completes the active tracker.

## Design rule

Presentation belongs to callers. The pipeline only owns execution and checkpoint configuration, so CLI formatting and web tracking can evolve independently.
