"""Scoped capability policy for agent tool access.

This module deliberately contains policy only. Concrete source, execution, and
artifact adapters can be added behind these capability names later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Set

from .contracts import AgentCapabilityManifest


CAPABILITIES = {
    "literature.search",
    "literature.fetch",
    "dataset.catalog",
    "dataset.download",
    "artifact.read",
    "artifact.write",
    "code.generate",
    "code.execute",
    "analysis.statistics",
    "verification.replay",
    "verification.claims",
}


@dataclass(frozen=True)
class CapabilityDecision:
    """Auditable result of a capability request."""

    allowed: bool
    agent: str
    capability: str
    reason: str


@dataclass(frozen=True)
class SandboxCapabilityManifest:
    """Shared execution limits used by planning, debate, and engineering."""

    max_wall_clock_seconds: int = 120
    available_libraries: tuple[str, ...] = ("numpy", "scipy", "pandas", "sklearn")
    gpu_available: bool = False
    outbound_network: bool = False
    dataset_downloads: bool = False
    max_dataset_rows: int = 100_000
    max_training_epochs: int = 50
    max_samples: int = 100_000

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
            "available_libraries": list(self.available_libraries),
            "gpu_available": self.gpu_available,
            "outbound_network": self.outbound_network,
            "dataset_downloads": self.dataset_downloads,
            "max_dataset_rows": self.max_dataset_rows,
            "max_training_epochs": self.max_training_epochs,
            "max_samples": self.max_samples,
        }


SANDBOX_CAPABILITY_MANIFEST = SandboxCapabilityManifest()


def check_plan_feasibility(plan: Mapping[str, Any], manifest: SandboxCapabilityManifest = SANDBOX_CAPABILITY_MANIFEST) -> List[str]:
    """Return blocking reasons before an experiment contract is committed."""
    errors: List[str] = []
    plan_text = str(plan).lower()
    if not manifest.dataset_downloads and any(token in plan_text for token in ("download", "internet", "outbound", "yahoo finance", "wikipedia traffic", "uci")):
        errors.append("plan requires outbound dataset access, which the sandbox forbids")
    if not manifest.gpu_available and any(token in plan_text for token in ("gpu", "cuda", "large language model fine-tune", "deep neural network")):
        errors.append("plan requires GPU-scale execution, but no GPU is available")
    for index, experiment in enumerate(plan.get("experiments") or []):
        if not isinstance(experiment, Mapping):
            continue
        dataset = experiment.get("dataset") or {}
        rows = dataset.get("rows") or dataset.get("row_count") or dataset.get("size")
        if isinstance(rows, (int, float)) and rows > manifest.max_dataset_rows:
            errors.append(f"experiment[{index}] dataset exceeds max rows ({manifest.max_dataset_rows})")
        epochs = experiment.get("epochs") or experiment.get("training_epochs")
        if isinstance(epochs, (int, float)) and epochs > manifest.max_training_epochs:
            errors.append(f"experiment[{index}] exceeds max training epochs ({manifest.max_training_epochs})")
        samples = experiment.get("samples") or experiment.get("n_samples")
        if isinstance(samples, (int, float)) and samples > manifest.max_samples:
            errors.append(f"experiment[{index}] exceeds max samples ({manifest.max_samples})")
    return errors


def manifest_for(
    agent: str,
    role: str,
    allowed: Iterable[str],
    forbidden: Iterable[str] = (),
    inputs: Iterable[str] = (),
    outputs: Iterable[str] = (),
    can_mutate: Iterable[str] = (),
) -> AgentCapabilityManifest:
    """Build a normalized manifest and reject unknown capability names."""
    allowed_set = set(allowed)
    forbidden_set = set(forbidden)
    unknown = (allowed_set | forbidden_set) - CAPABILITIES
    if unknown:
        raise ValueError(f"Unknown capabilities: {sorted(unknown)}")
    overlap = allowed_set & forbidden_set
    if overlap:
        raise ValueError(f"Capabilities cannot be both allowed and forbidden: {sorted(overlap)}")
    return {
        "agent": agent,
        "role": role,
        "allowed_capabilities": sorted(allowed_set),
        "forbidden_capabilities": sorted(forbidden_set),
        "input_artifact_types": sorted(set(inputs)),
        "output_artifact_types": sorted(set(outputs)),
        "can_mutate": sorted(set(can_mutate)),
    }


def authorize(
    manifest: Mapping[str, object],
    capability: str,
) -> CapabilityDecision:
    """Authorize one operation using only the agent's declared manifest."""
    agent = str(manifest.get("agent") or "unknown")
    if capability not in CAPABILITIES:
        return CapabilityDecision(False, agent, capability, "unknown capability")
    allowed = set(manifest.get("allowed_capabilities") or [])
    forbidden = set(manifest.get("forbidden_capabilities") or [])
    if capability in forbidden:
        return CapabilityDecision(False, agent, capability, "explicitly forbidden")
    if capability not in allowed:
        return CapabilityDecision(False, agent, capability, "not declared by manifest")
    return CapabilityDecision(True, agent, capability, "declared by manifest")


DEFAULT_MANIFESTS = {
    "TopicHunterAgent": manifest_for(
        "TopicHunterAgent",
        "research discovery",
        {"literature.search", "literature.fetch", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics", "verification.claims"},
        {"research_domain", "prior_lessons"},
        {"topic_candidates"},
        {"topic_candidates"},
    ),
    "HypothesisDebateSystem": manifest_for(
        "HypothesisDebateSystem",
        "hypothesis review",
        {"literature.search", "literature.fetch", "artifact.read", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics"},
        {"topic_candidate", "prior_lessons"},
        {"debate_result"},
        {"debate_result"},
    ),
    "PlannerAgent": manifest_for(
        "PlannerAgent",
        "experiment planning",
        {"artifact.read", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics", "verification.claims"},
        {"topic", "debate_result"},
        {"plan"},
        {"plan"},
    ),
    "WriterAgent": manifest_for(
        "WriterAgent",
        "scientific writing",
        {"artifact.read", "artifact.write", "literature.search", "literature.fetch"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics"},
        {"topic", "plan", "evidence"},
        {"draft_section"},
        {"draft_section"},
    ),
    "SupervisorAgent": manifest_for(
        "SupervisorAgent",
        "quality supervision",
        {"artifact.read", "verification.claims", "literature.search", "literature.fetch"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics"},
        {"draft_section", "evidence"},
        {"supervisor_feedback"},
        {"supervisor_feedback"},
    ),
    "MetaAgent": manifest_for(
        "MetaAgent",
        "workflow evaluation",
        {"artifact.read", "artifact.write", "verification.claims"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics"},
        {"workflow_state", "feedback"},
        {"meta_feedback"},
        {"meta_feedback"},
    ),
    "EditorAgent": manifest_for(
        "EditorAgent",
        "manuscript assembly",
        {"artifact.read", "artifact.write", "verification.claims", "literature.fetch"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics"},
        {"draft_sections", "verification_findings", "citations"},
        {"final_paper", "latex_output"},
        {"final_paper", "latex_output"},
    ),
    "EngineerAgent": manifest_for(
        "EngineerAgent",
        "implementation",
        {"code.generate", "artifact.read", "artifact.write"},
        {"dataset.download", "code.execute", "analysis.statistics", "verification.claims"},
        {"plan", "dataset_spec"},
        {"code"},
        {"code"},
    ),
    "DataAgent": manifest_for(
        "DataAgent",
        "data stewardship",
        {"literature.search", "literature.fetch", "dataset.catalog", "dataset.download", "artifact.write"},
        {"code.generate", "code.execute", "analysis.statistics", "verification.claims"},
        {"research_question", "dataset_spec"},
        {"dataset"},
        {"dataset"},
    ),
    "ExecutionAgent": manifest_for(
        "ExecutionAgent",
        "experiment execution",
        {"artifact.read", "code.execute", "artifact.write"},
        {"code.generate", "dataset.download", "analysis.statistics", "verification.claims"},
        {"code", "dataset", "execution_request"},
        {"execution"},
        {"execution"},
    ),
    "AnalysisAgent": manifest_for(
        "AnalysisAgent",
        "statistical analysis",
        {"artifact.read", "analysis.statistics", "artifact.write"},
        {"code.generate", "code.execute", "dataset.download", "verification.claims"},
        {"execution", "analysis_plan"},
        {"statistical_report"},
        {"statistical_report"},
    ),
    "VerificationAgent": manifest_for(
        "VerificationAgent",
        "independent verification",
        {"artifact.read", "verification.replay", "verification.claims", "artifact.write"},
        {"code.generate", "dataset.download", "analysis.statistics"},
        {"dataset", "code", "execution", "statistical_report", "claim"},
        {"verification_finding"},
        {"verification_finding"},
    ),
}
