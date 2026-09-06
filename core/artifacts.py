"""Filesystem artifact serialization for completed research runs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from .config import config
from .ports import ArtifactPort
from .state import ResearchState

logger = logging.getLogger(__name__)


class FilesystemArtifactStore(ArtifactPort):
    """Store named artifacts below one controlled filesystem root."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, content: Any) -> str:
        target = (self.root / name).resolve()
        if self.root not in target.parents:
            raise ValueError("Artifact path must remain inside the configured root")
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        elif isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            with target.open("w", encoding="utf-8") as handle:
                json.dump(content, handle, indent=2, default=str)
        return str(target)


def save_results(state: ResearchState, output_dir: Optional[str] = None) -> None:
    """Write the final paper, plan, and operator-facing run summary."""
    try:
        target_dir = output_dir or config.output_dir
        os.makedirs(target_dir, exist_ok=True)

        if hasattr(state, "value"):
            state = state.value
        elif hasattr(state, "__dict__"):
            state = state.__dict__

        if state.get("terminal_error") or state.get("evidence_gate", {}).get("terminal"):
            dossier = save_failure_dossier(state, target_dir)
            logger.warning("Run stopped before release; failure dossier saved to %s", dossier)
            return

        if state.get("latex_output"):
            latex_file = os.path.join(target_dir, "paper_output.tex")
            with open(latex_file, "w", encoding="utf-8") as handle:
                handle.write(state["latex_output"])
            logger.info("LaTeX output saved to %s", latex_file)

        if state.get("plan"):
            import yaml

            plan_file = os.path.join(target_dir, "plan.yaml")
            with open(plan_file, "w", encoding="utf-8") as handle:
                yaml.dump(state["plan"], handle, default_flow_style=False)
            logger.info("Plan saved to %s", plan_file)

        summary = {
            "iteration": state.get("iteration", 0),
            "selected_topic": state.get("selected_topic"),
            "sections_written": list(state.get("draft_sections", {}).keys()),
            "supervisor_scores": state.get("supervisor_scores", {}),
            "experiments_run": list(state.get("engineer_outputs", {}).keys()),
            "meta_feedback": state.get("meta_feedback", []),
        }
        summary_file = os.path.join(target_dir, "research_summary.json")
        with open(summary_file, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Research summary saved to %s", summary_file)
    except Exception as exc:
        logger.error("Error saving results: %s", exc)
        print(f"Warning: Could not save results: {exc}")


def save_failure_dossier(state: ResearchState, output_dir: str) -> str:
    """Persist an operator-readable failure report without releasing a paper."""
    from .run_log import build_run_summary, get_tracker
    from .research_db import research_db

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    summary = build_run_summary(state, error=state.get("terminal_error"))
    dossier = {
        "status": "failed",
        "run_id": state.get("run_id") or summary.get("run_id"),
        "terminal_error": state.get("terminal_error"),
        "evidence_gate": state.get("evidence_gate", {}),
        "experiment_contracts": state.get("experiment_contracts", {}),
        "technical_failures": state.get("technical_failures", {}),
        "engineer_outputs": state.get("engineer_outputs", {}),
        "execution_artifacts": state.get("execution_artifacts", {}),
        "verification_findings": state.get("verification_findings", []),
        "summary": summary,
    }
    dossier_path = target_dir / "failure_dossier.json"
    dossier_path.write_text(json.dumps(dossier, indent=2, default=str), encoding="utf-8")
    summary_path = target_dir / "research_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    tracker = get_tracker()
    if tracker:
        research_db.record_artifact(
            tracker.run_id,
            "failure_dossier",
            str(dossier_path),
            {"reason": state.get("terminal_error"), "gate": state.get("evidence_gate", {})},
        )
        research_db.update_run_summary(
            tracker.run_id,
            {"status": "failed", "terminal_error": state.get("terminal_error"), "failure_dossier": str(dossier_path)},
        )
    return str(dossier_path)
