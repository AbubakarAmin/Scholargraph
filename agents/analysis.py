"""Independent statistical analysis of execution artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import numpy as np
from scipy import stats

from core.capabilities import DEFAULT_MANIFESTS
from core.context import RunContext, get_active_context
from core.contracts import AnalysisPlan, ExecutionArtifact, StatisticalReport
from core.verification import preregister_power


class AnalysisAgent:
    """Analyze stored results without generating code or changing experiments."""

    capability_manifest = DEFAULT_MANIFESTS["AnalysisAgent"]

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()

    def analyze(
        self,
        execution_artifacts: Mapping[str, ExecutionArtifact],
        plan: AnalysisPlan,
    ) -> StatisticalReport:
        warnings: list[str] = []
        metrics: dict[str, Any] = {}
        for name, artifact in execution_artifacts.items():
            if artifact.get("status") != "completed":
                warnings.append(f"execution failed: {name}")
                continue
            aggregate = (artifact.get("seed_results") or {}).get("aggregate_metrics") or {}
            metrics[name] = {
                metric: self._summarize(values, warnings, f"{name}.{metric}")
                for metric, values in aggregate.items()
            }

        comparisons = self._compare(metrics, plan, warnings)
        self._apply_multiple_comparison_correction(comparisons, plan)
        passed = bool(metrics) and not any(
            "missing" in warning or "failed" in warning or "underpowered" in warning
            for warning in warnings
        )
        power_plan = None
        if plan.get("require_power_analysis") and not plan.get("planned_effect_size"):
            warnings.append("missing prospective power preregistration")
            passed = False
        elif plan.get("planned_effect_size"):
            power_plan = preregister_power(
                float(plan["planned_effect_size"]),
                float(plan.get("alpha", 0.05)),
                float(plan.get("target_power", 0.8)),
            )
        now = datetime.now(timezone.utc).isoformat()
        payload = {"metrics": metrics, "comparisons": comparisons, "warnings": warnings}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return {
            "analysis_plan": plan,
            "metrics": metrics,
            "comparisons": comparisons,
            "power_plan": power_plan,
            "warnings": warnings,
            "passed": passed,
            "provenance": {
                "artifact_type": "statistical_report",
                "producer": "AnalysisAgent",
                "content_hash": digest,
                "created_at": now,
                "status": "complete" if passed else "needs_review",
                "limitations": ["Statistical conclusions are limited by the supplied seeds and raw outputs."],
            },
        }

    @staticmethod
    def _summarize(raw: Any, warnings: list[str], label: str) -> dict[str, Any]:
        values = raw.get("values") if isinstance(raw, dict) else None
        if not values:
            warnings.append(f"missing raw values: {label}")
            return {"mean": raw.get("mean") if isinstance(raw, dict) else raw, "n": 0}
        sample = np.asarray(values, dtype=float)
        n = int(sample.size)
        mean = float(sample.mean())
        std = float(sample.std(ddof=1)) if n > 1 else 0.0
        if n < 3:
            warnings.append(f"insufficient seeds for uncertainty estimate: {label} (n={n})")
            interval = [mean, mean]
        else:
            margin = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
            interval = [mean - margin, mean + margin]
        return {"mean": mean, "std": std, "n": n, "confidence_interval_95": interval, "values": values}

    @staticmethod
    def _compare(metrics: dict[str, Any], plan: AnalysisPlan, warnings: list[str]) -> list[dict[str, Any]]:
        metric_name = plan.get("primary_metric")
        if not metric_name:
            warnings.append("missing primary metric")
            return []
        names = list(metrics)
        if len(names) < 2:
            warnings.append("missing comparison artifact")
            return []
        baseline_name = names[0]
        baseline = metrics[baseline_name].get(metric_name)
        comparisons = []
        for name in names[1:]:
            candidate = metrics[name].get(metric_name)
            if not baseline or not candidate:
                warnings.append(f"missing comparison metric: {name}.{metric_name}")
                continue
            base_values = np.asarray(baseline.get("values", []), dtype=float)
            candidate_values = np.asarray(candidate.get("values", []), dtype=float)
            if len(base_values) < 3 or len(candidate_values) < 3:
                warnings.append(f"underpowered comparison: {baseline_name} vs {name} requires at least 3 seeds")
                continue
            test = stats.ttest_ind(candidate_values, base_values, equal_var=False)
            pooled = np.sqrt((candidate_values.var(ddof=1) + base_values.var(ddof=1)) / 2)
            effect = float((candidate_values.mean() - base_values.mean()) / pooled) if pooled else 0.0
            alpha = float(plan.get("alpha", 0.05))
            target_power = float(plan.get("target_power", 0.8))
            z_alpha = float(stats.norm.ppf(1 - alpha / 2))
            z_power = float(stats.norm.ppf(target_power))
            required_n = int(np.ceil(2 * ((z_alpha + z_power) / max(abs(effect), 1e-9)) ** 2))
            power_estimate = float(stats.norm.cdf(abs(effect) * np.sqrt(len(base_values) / 2) - z_alpha))
            if len(base_values) < required_n or len(candidate_values) < required_n:
                warnings.append(f"underpowered comparison: {baseline_name} vs {name} requires n>={required_n} per group")
            comparisons.append({
                "candidate": name,
                "baseline": baseline_name,
                "metric": metric_name,
                "difference": float(candidate_values.mean() - base_values.mean()),
                "p_value": float(test.pvalue),
                "cohens_d": effect,
                "test": "Welch t-test",
                "alpha": alpha,
                "target_power": target_power,
                "required_n_per_group": required_n,
                "observed_power_estimate": power_estimate,
            })
        return comparisons

    @staticmethod
    def _apply_multiple_comparison_correction(comparisons: list[dict[str, Any]], plan: AnalysisPlan) -> None:
        """Attach adjusted p-values; never leave a multi-test report ambiguous."""
        if not comparisons:
            return
        policy = str(plan.get("multiple_comparison_policy") or "bonferroni").lower()
        p_values = [float(item["p_value"]) for item in comparisons]
        if policy in {"none", "uncorrected"}:
            adjusted = p_values
        elif policy in {"holm", "holm-bonferroni"}:
            ordered = sorted(enumerate(p_values), key=lambda pair: pair[1])
            adjusted = [0.0] * len(p_values)
            for rank, (index, value) in enumerate(ordered):
                adjusted[index] = min(1.0, value * (len(p_values) - rank))
        else:
            policy = "bonferroni"
            adjusted = [min(1.0, value * len(p_values)) for value in p_values]
        for item, value in zip(comparisons, adjusted):
            item["p_value_adjusted"] = float(value)
            item["multiple_comparison_policy"] = policy
            item["significant_at_0.05"] = bool(value < 0.05)
