"""Deterministic gates for immutable experiments and evidence handoffs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .contracts import EvidenceGateDecision, ExperimentContract, ExperimentSpec


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_experiments(experiments: Any) -> List[str]:
    """Return deterministic validation errors for a plan's experiment list."""
    errors: List[str] = []
    if not isinstance(experiments, list) or not experiments:
        return ["plan.experiments must be a non-empty list"]
    names = set()
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            errors.append(f"experiment[{index}] must be an object")
            continue
        name = experiment.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"experiment[{index}] must have a non-empty name")
            continue
        if name in names:
            errors.append(f"duplicate experiment name: {name}")
        names.add(name)
        if not experiment.get("evaluation_metrics"):
            errors.append(f"{name} must declare evaluation_metrics")
        variants = experiment.get("variants") or experiment.get("alternatives") or []
        for variant_index, variant in enumerate(variants):
            if not isinstance(variant, dict):
                errors.append(f"{name} variant[{variant_index}] must be an object")
    return errors


def build_contract(experiment: ExperimentSpec, dataset: Mapping[str, Any] | None = None) -> ExperimentContract:
    """Create the immutable snapshot used by all repair and validation attempts."""
    contract: ExperimentContract = {
        "experiment_name": str(experiment.get("name") or ""),
        "hypothesis": str(experiment.get("falsifiable_prediction") or experiment.get("description") or ""),
        "dataset": dict(dataset or experiment.get("dataset") or {}),
        "requirements": {
            "type": experiment.get("type"),
            "methodology": experiment.get("methodology"),
            "claimed_components": experiment.get("claimed_components") or [],
        },
        "baselines": list(experiment.get("baselines") or experiment.get("baseline_comparison") or []),
        "evaluation_metrics": list(experiment.get("evaluation_metrics") or []),
        "split_policy": str(experiment.get("split_policy") or ""),
        "seeds": list(experiment.get("seeds") or []),
        "stopping_rule": str(experiment.get("stopping_rule") or ""),
        "analysis_protocol": {
            "statistical_test": experiment.get("statistical_test"),
            "analysis_plan": experiment.get("analysis_plan") or {},
        },
    }
    contract["contract_hash"] = _stable_hash(contract)
    return contract


def contract_matches(contract: Mapping[str, Any], experiment: Mapping[str, Any]) -> bool:
    expected = build_contract(experiment, contract.get("dataset"))
    return expected.get("contract_hash") == contract.get("contract_hash")


def validate_dataset_identity(
    experiments: Iterable[Mapping[str, Any]],
    data_artifacts: Mapping[str, Mapping[str, Any]] | None,
) -> List[str]:
    """Ensure declared dataset names and hashes resolve to validated artifacts."""
    artifacts = data_artifacts or {}
    errors: List[str] = []
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            continue
        name = experiment.get("name") or "unnamed"
        declared = experiment.get("dataset") or {}
        if not declared:
            continue
        dataset_name = declared.get("name") or declared.get("id")
        if not dataset_name:
            errors.append(f"{name} declares a dataset without a name or id")
            continue
        artifact = artifacts.get(dataset_name)
        if not artifact:
            errors.append(f"{name} dataset artifact not found: {dataset_name}")
            continue
        if artifact.get("validation", {}).get("passed") is False:
            errors.append(f"{name} dataset failed validation: {dataset_name}")
        declared_hash = declared.get("content_hash") or declared.get("hash")
        actual_hash = artifact.get("content_hash") or artifact.get("provenance", {}).get("content_hash")
        if declared_hash and actual_hash and declared_hash != actual_hash:
            errors.append(f"{name} dataset hash changed: {dataset_name}")
    return errors


def gate_engineering_outputs(
    plan: Mapping[str, Any] | None,
    outputs: Mapping[str, Mapping[str, Any]] | None,
    contracts: Mapping[str, Mapping[str, Any]] | None,
) -> EvidenceGateDecision:
    """Decide whether engineering may hand off to independent validation."""
    experiments = (plan or {}).get("experiments") or []
    errors = validate_experiments(experiments)
    if errors:
        return {
            "allowed": False,
            "terminal": True,
            "reason_code": "invalid_experiment_plan",
            "message": "; ".join(errors),
        }
    outputs = outputs or {}
    contracts = contracts or {}
    names = [experiment["name"] for experiment in experiments if isinstance(experiment, dict)]
    missing = [name for name in names if name not in outputs]
    if missing:
        return {
            "allowed": False,
            "terminal": True,
            "reason_code": "missing_experiment_output",
            "message": f"No engineering output for: {', '.join(missing)}",
            "experiment_names": names,
        }
    for name in names:
        output = outputs[name]
        if not output.get("success"):
            return {
                "allowed": False,
                "terminal": True,
                "reason_code": "technical_execution_failure",
                "message": f"Experiment {name} did not complete: {output.get('error') or 'unknown error'}",
                "experiment_names": names,
            }
        contract = contracts.get(name) or {}
        if not contract.get("contract_hash") or output.get("contract_hash") != contract.get("contract_hash"):
            return {
                "allowed": False,
                "terminal": True,
                "reason_code": "contract_provenance_mismatch",
                "message": f"Experiment {name} is not tied to its committed contract",
                "experiment_names": names,
            }
        raw_path = output.get("raw_results_path")
        if not raw_path or not Path(raw_path).is_file():
            return {
                "allowed": False,
                "terminal": True,
                "reason_code": "missing_raw_results",
                "message": f"Experiment {name} has no raw results artifact",
                "experiment_names": names,
            }
        if not output.get("aggregate_metrics") and not output.get("results", {}).get("metrics"):
            return {
                "allowed": False,
                "terminal": True,
                "reason_code": "missing_metrics",
                "message": f"Experiment {name} produced no metrics",
                "experiment_names": names,
            }
    return {
        "allowed": True,
        "terminal": False,
        "reason_code": "engineering_complete",
        "message": "All committed experiments completed with raw metrics and provenance",
        "experiment_names": names,
        "contract_hashes": {name: contracts[name]["contract_hash"] for name in names},
    }
