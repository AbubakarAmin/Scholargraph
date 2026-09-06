"""Scoped capability policy for agent tool access.

This module deliberately contains policy only. Concrete source, execution, and
artifact adapters can be added behind these capability names later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Set

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
