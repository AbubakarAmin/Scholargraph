"""Shared state contract for the ScholarGraph workflow."""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from .contracts import DatasetArtifact, ExecutionArtifact, ExperimentContract, ExperimentOutput, LiteratureContext, Plan, StatisticalReport, Topic, VerificationReport


class ResearchState(TypedDict):
    """Mutable state passed between LangGraph workflow nodes."""

    mode: Union[Literal["full_research"], Literal["qa"]]
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
    data_artifacts: Dict[str, DatasetArtifact]
    data_validation: VerificationReport
    execution_artifacts: Dict[str, ExecutionArtifact]
    analysis_reports: Dict[str, StatisticalReport]
    verification_findings: List[Dict[str, Any]]
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
    experiment_contracts: Dict[str, ExperimentContract]
    experiment_outcomes: Dict[str, str]
    technical_failures: Dict[str, Dict[str, Any]]
    evidence_gate: Dict[str, Any]
    human_approved: bool
    outcome_calibration: Dict[str, Any]
    literature_context: Optional[LiteratureContext]
    qa_answer: Optional[Dict[str, Any]]
    qa_citation_verification: Optional[Dict[str, Any]]
    user_query: Optional[str]


def initialize_state(mode: str = "full_research") -> ResearchState:
    """Return a fresh state for a new research run."""
    return ResearchState(
        mode=mode,
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
        data_artifacts={},
        data_validation={},
        execution_artifacts={},
        analysis_reports={},
        verification_findings=[],
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
        experiment_contracts={},
        experiment_outcomes={},
        technical_failures={},
        evidence_gate={},
        human_approved=False,
        outcome_calibration={},
        literature_context=None,
        qa_answer=None,
        qa_citation_verification=None,
        user_query=None,
    )