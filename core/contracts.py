"""Typed handoff contracts shared by agents and workflow state."""

from typing import Any, Dict, List, TypedDict


class ArtifactProvenance(TypedDict, total=False):
    """Traceability metadata shared by every research artifact."""

    artifact_id: str
    run_id: str
    producer: str
    artifact_type: str
    content_hash: str
    created_at: str
    source: str
    parent_artifacts: List[str]
    environment: Dict[str, Any]
    limitations: List[str]
    status: str


class ResearchQuestion(TypedDict, total=False):
    question_id: str
    question: str
    hypothesis: str
    falsifiable_prediction: str
    null_hypothesis: str
    primary_metric: str
    status: str


class DatasetSpec(TypedDict, total=False):
    name: str
    source: str
    version: str
    license: str
    task: str
    features: List[str]
    target: str
    split_policy: str
    leakage_controls: List[str]
    access_policy: str


class SourceArtifact(TypedDict, total=False):
    provenance: ArtifactProvenance
    source: str
    url: str
    retrieved_at: str
    status: str
    response_hash: str
    content: Dict[str, Any]
    warnings: List[str]


class DatasetArtifact(TypedDict, total=False):
    provenance: ArtifactProvenance
    spec: DatasetSpec
    location: str
    content_hash: str
    schema: Dict[str, Any]
    row_count: int
    validation: "VerificationReport"


class CodeArtifact(TypedDict, total=False):
    provenance: ArtifactProvenance
    experiment_name: str
    source_code: str
    language: str
    dependencies: List[str]
    entrypoint: str
    validation: "VerificationReport"


class ExecutionRequest(TypedDict, total=False):
    experiment_name: str
    code_artifact_id: str
    dataset_artifact_ids: List[str]
    seeds: List[int]
    timeout_seconds: int
    resource_limits: Dict[str, Any]


class ExecutionArtifact(TypedDict, total=False):
    provenance: ArtifactProvenance
    request: ExecutionRequest
    raw_results_path: str
    environment: Dict[str, Any]
    seed_results: Dict[str, Any]
    status: str
    error: str
    content_hash: str


class AnalysisPlan(TypedDict, total=False):
    primary_metric: str
    secondary_metrics: List[str]
    statistical_test: str
    effect_size: str
    confidence_level: float
    multiple_comparison_policy: str
    stopping_rule: str


class StatisticalReport(TypedDict, total=False):
    provenance: ArtifactProvenance
    analysis_plan: AnalysisPlan
    metrics: Dict[str, Any]
    comparisons: List[Dict[str, Any]]
    warnings: List[str]
    passed: bool


class Claim(TypedDict, total=False):
    claim_id: str
    text: str
    claim_type: str
    evidence_artifact_ids: List[str]
    status: str
    limitations: List[str]


class EvidenceBundle(TypedDict, total=False):
    claim: Claim
    dataset_artifacts: List[DatasetArtifact]
    code_artifacts: List[CodeArtifact]
    execution_artifacts: List[ExecutionArtifact]
    statistical_reports: List[StatisticalReport]
    verification_findings: List[Dict[str, Any]]


class VerificationFinding(TypedDict, total=False):
    finding_id: str
    severity: str
    check: str
    message: str
    artifact_ids: List[str]
    blocking: bool
    status: str


class AgentCapabilityManifest(TypedDict, total=False):
    agent: str
    role: str
    allowed_capabilities: List[str]
    forbidden_capabilities: List[str]
    input_artifact_types: List[str]
    output_artifact_types: List[str]
    can_mutate: List[str]


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
