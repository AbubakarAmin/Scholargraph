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
    workflow.add_node("writing_narrative", nodes["writing_narrative"])
    workflow.add_node("engineering", nodes["engineering"])
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
    workflow.add_edge("planning", "writing_narrative")
    workflow.add_edge("writing_narrative", "engineering")

    workflow.add_conditional_edges(
        "engineering",
        lambda state: "planning" if state.get("current_phase") == "planning" else "writing_results",
        {"planning": "planning", "writing_results": "writing_results"},
    )
    workflow.add_conditional_edges(
        "writing_results",
        lambda state: "redraft" if state.get("current_phase") == "writing_results" else "supervision",
        {"redraft": "writing_results", "supervision": "supervision"},
    )
    workflow.add_conditional_edges(
        "supervision",
        lambda state: "editing" if state["current_phase"] == "editing" else "meta_evaluation",
        {"editing": "editing", "meta_evaluation": "meta_evaluation"},
    )
    workflow.add_conditional_edges(
        "meta_evaluation",
        nodes["should_continue"],
        {"continue": "writing_narrative", "end": END},
    )
    workflow.add_edge("editing", END)
    workflow.add_edge("reset", "topic_discovery")

    return workflow
