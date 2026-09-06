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
        passed = bool(metrics) and not any("missing" in warning or "failed" in warning for warning in warnings)
        now = datetime.now(timezone.utc).isoformat()
        payload = {"metrics": metrics, "comparisons": comparisons, "warnings": warnings}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return {
            "analysis_plan": plan,
            "metrics": metrics,
            "comparisons": comparisons,
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
            if len(base_values) < 2 or len(candidate_values) < 2:
                warnings.append(f"insufficient values for test: {baseline_name} vs {name}")
                continue
            test = stats.ttest_ind(candidate_values, base_values, equal_var=False)
            pooled = np.sqrt((candidate_values.var(ddof=1) + base_values.var(ddof=1)) / 2)
            effect = float((candidate_values.mean() - base_values.mean()) / pooled) if pooled else 0.0
            comparisons.append({
                "candidate": name,
                "baseline": baseline_name,
                "metric": metric_name,
                "difference": float(candidate_values.mean() - base_values.mean()),
                "p_value": float(test.pvalue),
                "cohens_d": effect,
                "test": "Welch t-test",
            })
        return comparisons
