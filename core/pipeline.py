"""Shared execution service for CLI and web research runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional

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
    ) -> None:
        self._graph_factory = graph_factory
        self._checkpointer_factory = checkpointer_factory
        self._recursion_limit = recursion_limit
        self.context = context or create_run_context()

    def run(
        self,
        initial_state: Optional[ResearchState],
        run_id: str,
        resume: bool = False,
        on_node: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
        finalize: Optional[Callable[[ResearchState], None]] = None,
    ) -> PipelineResult:
        """Execute the graph, finalize artifacts, and complete the run tracker."""
        last_state = initial_state or {}
        nodes_seen = 0
        try:
            for node_name, node_output in self.stream(initial_state, run_id, resume=resume):
                nodes_seen += 1
                last_state = node_output
                if on_node:
                    on_node(node_name, node_output)
            if finalize:
                finalize(last_state)
            if self.context.tracker:
                self.context.tracker.complete(
                    success=bool(last_state.get("latex_output"))
                    and not last_state.get("terminal_error")
                )
            return PipelineResult(last_state, nodes_seen)
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
            graph = self._graph_factory()
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
