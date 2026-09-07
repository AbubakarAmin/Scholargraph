"""LangGraph graph construction for the ScholarGraph workflow.

Node implementations remain injectable so the composition root can migrate
without changing the state-machine behavior or public entry points.
"""

from typing import Callable, Mapping

from langgraph.graph import END, StateGraph

from .state import ResearchState

Node = Callable[[ResearchState], ResearchState]


def create_research_graph(nodes: Mapping[str, Node]) -> StateGraph:
    """Build the research graph from its phase node implementations."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("topic_discovery", nodes["topic_discovery"])
    workflow.add_node("hypothesis_debate", nodes["hypothesis_debate"])
    workflow.add_node("planning", nodes["planning"])
    workflow.add_node("data_validation", nodes["data_validation"])
    workflow.add_node("writing_narrative", nodes["writing_narrative"])
    workflow.add_node("engineering", nodes["engineering"])
    workflow.add_node("independent_validation", nodes["independent_validation"])
    workflow.add_node("writing_results", nodes["writing_results"])
    workflow.add_node("supervision", nodes["supervision"])
    workflow.add_node("meta_evaluation", nodes["meta_evaluation"])
    workflow.add_node("editing", nodes["editing"])
    workflow.add_node("reset", nodes["reset"])

    workflow.set_entry_point("topic_discovery")

    workflow.add_conditional_edges(
        "topic_discovery",
        nodes["should_reset"],
        {"reset": "reset", "continue": "hypothesis_debate", "end": END},
    )
    workflow.add_conditional_edges(
        "hypothesis_debate",
        nodes["should_reset"],
        {"reset": "reset", "continue": "planning", "end": END},
    )

    def _after_planning(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("should_reset"):
            return "reset"
        return "continue"

    workflow.add_conditional_edges(
        "planning",
        _after_planning,
        {"continue": "data_validation", "reset": "reset", "end": END},
    )

    def _after_data_validation(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("should_reset"):
            return "reset"
        return "continue"

    workflow.add_conditional_edges(
        "data_validation",
        _after_data_validation,
        {"continue": "writing_narrative", "reset": "reset", "end": END},
    )

    def _after_writing_narrative(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("should_reset"):
            return "reset"
        return "continue"

    workflow.add_conditional_edges(
        "writing_narrative",
        _after_writing_narrative,
        {"continue": "engineering", "reset": "reset", "end": END},
    )

    def _after_engineering(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("current_phase") == "planning":
            return "planning"
        if state.get("should_reset"):
            return "reset"
        return "continue"

    workflow.add_conditional_edges(
        "engineering",
        _after_engineering,
        {"planning": "planning", "continue": "independent_validation", "reset": "reset", "end": END},
    )

    def _after_independent_validation(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("should_reset"):
            return "reset"
        return "continue"

    workflow.add_conditional_edges(
        "independent_validation",
        _after_independent_validation,
        {"continue": "writing_results", "reset": "reset", "end": END},
    )

    def _after_writing_results(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("current_phase") == "writing_results":
            return "redraft"
        if state.get("should_reset"):
            return "reset"
        return "continue"

    workflow.add_conditional_edges(
        "writing_results",
        _after_writing_results,
        {"redraft": "writing_results", "continue": "supervision", "reset": "reset", "end": END},
    )

    def _after_supervision(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("current_phase") == "editing":
            return "editing"
        if state.get("should_reset"):
            return "reset"
        return "meta_evaluation"

    workflow.add_conditional_edges(
        "supervision",
        _after_supervision,
        {"editing": "editing", "meta_evaluation": "meta_evaluation", "reset": "reset", "end": END},
    )
    workflow.add_conditional_edges(
        "meta_evaluation",
        nodes["should_continue"],
        {"continue": "writing_narrative", "end": END},
    )
    workflow.add_edge("editing", END)
    workflow.add_edge("reset", "topic_discovery")

    return workflow
