"""Shared state contract for the ScholarGraph workflow."""

from typing import Any, Dict, List, Optional, TypedDict

from .contracts import ExperimentOutput, Plan, Topic, VerificationReport


class ResearchState(TypedDict):
    """Mutable state passed between LangGraph workflow nodes."""

    iteration: int
    current_phase: str
    should_reset: bool
    should_continue: bool
    error_count: int
    topics: List[Topic]
    selected_topic: Optional[Topic]
    debate_results: List[Dict[str, Any]]
    hypothesis_passed: bool
    plan: Optional[Plan]
    draft_sections: Dict[str, str]
    current_section: Optional[str]
    engineer_outputs: Dict[str, ExperimentOutput]
    supervisor_scores: Dict[str, float]
    supervisor_feedback: Dict[str, str]
    meta_feedback: List[str]
    final_paper: Optional[Dict[str, Any]]
    latex_output: Optional[str]
    plan_revision_requests: List[Dict[str, Any]]
    run_id: Optional[str]
    results_redraft_count: int
    results_verification: Dict[str, VerificationReport]
    reproducibility: VerificationReport
    terminal_error: Optional[str]


def initialize_state() -> ResearchState:
    """Return a fresh state for a new research run."""
    return ResearchState(
        iteration=0,
        current_phase="topic_discovery",
        should_reset=False,
        should_continue=True,
        error_count=0,
        topics=[],
        selected_topic=None,
        debate_results=[],
        hypothesis_passed=False,
        plan=None,
        draft_sections={},
        current_section=None,
        engineer_outputs={},
        supervisor_scores={},
        supervisor_feedback={},
        meta_feedback=[],
        final_paper=None,
        latex_output=None,
        plan_revision_requests=[],
        run_id=None,
        results_redraft_count=0,
        results_verification={},
        reproducibility={},
        terminal_error=None,
    )