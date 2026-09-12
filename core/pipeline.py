"""Shared execution service for CLI and web research runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from .context import RunContext, activate_context, create_run_context, reset_context
from .state import ResearchState


@dataclass
class PipelineResult:
    """Final state and node count produced by one pipeline execution."""

    state: ResearchState
    nodes_seen: int


class ResearchPipeline:
    """Compile and stream a research graph with durable checkpoint settings."""

    def __init__(
        self,
        graph_factory: Callable[[], Any],
        checkpointer_factory: Callable[[], Any],
        recursion_limit: int = 1000,
        context: Optional[RunContext] = None,
        mode_graphs: Optional[Dict[str, Callable[[], Any]]] = None,
    ) -> None:
        self._graph_factory = graph_factory
        self._checkpointer_factory = checkpointer_factory
        self._recursion_limit = recursion_limit
        self.context = context or create_run_context()
        self._mode_graphs = mode_graphs or {}

    def _resolve_graph_factory(self, state: Optional[ResearchState]) -> Callable[[], Any]:
        """Return the graph factory for the run's mode, falling back to default."""
        if state:
            mode = state.get("mode", "full_research")
            if mode in self._mode_graphs:
                return self._mode_graphs[mode]
        return self._graph_factory

    def run(
        self,
        initial_state: Optional[ResearchState],
        run_id: str,
        resume: bool = False,
        on_node: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
        finalize: Optional[Callable[[ResearchState], None]] = None,
    ) -> PipelineResult:
        """Execute the graph, finalize artifacts, and complete the run tracker."""
        full_state = dict(initial_state) if initial_state else {}
        nodes_seen = 0
        try:
            for node_name, node_output in self.stream(initial_state, run_id, resume=resume):
                nodes_seen += 1
                full_state.update(node_output)
                if on_node:
                    on_node(node_name, full_state)
            if finalize:
                finalize(full_state)
            if self.context.tracker:
                self.context.tracker.complete(
                    success=bool(full_state.get("latex_output") or full_state.get("qa_answer"))
                    and not full_state.get("terminal_error")
                )
            return PipelineResult(full_state, nodes_seen)
        except Exception:
            if self.context.tracker:
                self.context.tracker.complete(success=False)
            raise

    def stream(
        self,
        initial_state: Optional[ResearchState],
        run_id: str,
        resume: bool = False,
    ) -> Iterator[tuple[str, Mapping[str, Any]]]:
        """Yield each workflow node output for a new or resumed run."""
        self.context.run_id = run_id
        token = activate_context(self.context)
        try:
            graph_factory = self._resolve_graph_factory(initial_state)
            graph = graph_factory()
            app = graph.compile(checkpointer=self._checkpointer_factory())
            config = {
                "configurable": {"thread_id": run_id},
                "recursion_limit": self._recursion_limit,
            }
            source = None if resume else initial_state
            for event in app.stream(source, config):
                for node_name, node_output in event.items():
                    if node_name != "__end__":
                        yield node_name, node_output
        finally:
            reset_context(token)
