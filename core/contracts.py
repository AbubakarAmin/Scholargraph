"""Typed handoff contracts shared by agents and workflow state."""

from typing import Any, Dict, List, TypedDict


class Topic(TypedDict, total=False):
    title: str
    description: str
    rationale: str
    impact: str
    feasibility: float
    score: float
    rank: int
    keywords: List[str]
    anchor_paper: str
    dataset_plan: str


class PlanSection(TypedDict, total=False):
    name: str
    content_requirements: str
    key_points: List[str]
    expected_length: str
    dependencies: List[str]


class Contribution(TypedDict, total=False):
    claim: str
    falsifiable_prediction: str
    statistical_test: str
    components: List[str]


Dependency = TypedDict(
    "Dependency",
    {"from": str, "to": str, "type": str, "description": str},
    total=False,
)


class RevisionRequest(TypedDict, total=False):
    reason: str
    detail: str
    experiment: str


class Timeline(TypedDict, total=False):
    phases: List[Dict[str, Any]]
    total_duration: str


class ExperimentSpec(TypedDict, total=False):
    name: str
    type: str
    description: str
    methodology: str
    baselines: List[str]
    variants: List[Dict[str, Any]]
    alternatives: List[Dict[str, Any]]
    falsifiable_prediction: str
    statistical_test: str
    claimed_components: List[str]
    evaluation_metrics: List[str]


class Plan(TypedDict, total=False):
    title: str
    topic: str
    domain: str
    sections: List[PlanSection]
    research_questions: List[str]
    methodology: str
    contributions: List[Contribution]
    expected_contributions: List[str]
    experiments: List[ExperimentSpec]
    dependencies: List[Dependency]
    timeline: Timeline
    revision_history: List[Dict[str, Any]]


class ExperimentOutput(TypedDict, total=False):
    experiment_name: str
    success: bool
    code: str
    aggregate_metrics: Dict[str, Any]
    results: Dict[str, Any]
    validation: Dict[str, Any]
    ablation: Dict[str, Any]
    raw_results_path: str
    error: str
    timestamp: str


class VerificationReport(TypedDict, total=False):
    passed: bool
    score: float
    note: str
    mismatches: List[str]
    checks: Dict[str, Any]


class FeasibilityReport(TypedDict, total=False):
    ok: bool
    reasons: List[str]


class CodeClaimReport(TypedDict, total=False):
    consistent: bool
    score: float
    notes: List[str]


class MetaDashboard(TypedDict, total=False):
    iteration: int
    phase: str
    rejected_topics: int
    debate_rounds: int
    sections_bounced: int
    pivots: int
    refines: int
    plan_revisions: int
    hard_check_fails: int
    avg_supervisor: float
    lessons_available: int


class MetaChatResult(TypedDict, total=False):
    reply: str
    control: str
    control_reason: str
    summary: Dict[str, Any]


class Paper(TypedDict, total=False):
    topic: Topic
    sections: Dict[str, str]
    plan: Plan
    engineer_outputs: Dict[str, ExperimentOutput]
    bibliography: str
    bib_map: Dict[str, str]
    companion_repo: Dict[str, str]
    debate_results: List[Any]
    timestamp: str
