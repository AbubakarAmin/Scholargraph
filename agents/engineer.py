"""
EngineerAgent — verifiable, self-healing experiment runner.
Sandbox lockdown, multi-seed, PIVOT/REFINE, ablations, code-claim checks.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import log_agent_action, parse_json_from_llm
from core.llm import call_llm, generate_embedding
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.memory import memory
from core.sandbox import execute_sandboxed, run_multi_seed, run_known_answer_check, validate_code
from core.run_log import get_tracker, CrossRunMemory, emit_event
from core.research_db import research_db
from core.contracts import CodeClaimReport, ExperimentOutput, ExperimentSpec, RevisionRequest
from core.known_answers import fixture_for


class EngineerAgent:
    """Runs experiments with hard sandboxing and recovery loops."""

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()
        runtime_config = self.context.config if self.context else config
        self.ledger = self.context.research_db if self.context else research_db
        self.vector_memory = self.context.memory if self.context else memory
        self.output_dir = runtime_config.output_dir
        self.raw_dir = runtime_config.raw_results_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        self._plan_revision_requests: List[Dict[str, Any]] = []

    @property
    def runtime_config(self):
        return self.context.config if self.context else config

    def _progress(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit live tactical progress for Arena / Meta chat."""
        tracker = get_tracker()
        run_id = tracker.run_id if tracker else None
        emit_event(event_type, data, run_id=run_id, agent="EngineerAgent")
        if tracker:
            tracker.scratch("EngineerAgent", event_type, data)
            tracker.message(f"{event_type}: {data.get('experiment') or data.get('name') or data.get('decision') or ''}")

    def request_plan_revision(self, reason: str, experiment: ExperimentSpec, detail: str = "") -> RevisionRequest:
        """First-class path back to Planner when the plan is wrong."""
        req = {
            "reason": reason,
            "experiment": experiment.get("name"),
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
        }
        self._plan_revision_requests.append(req)
        CrossRunMemory().record_plan_revision(reason, meta=req)
        tracker = get_tracker()
        if tracker:
            tracker.bump("plan_revisions")
            tracker.scratch("EngineerAgent", "plan_revision_request", req)
        log_agent_action("EngineerAgent", "request_plan_revision", req)
        return req

    def consume_plan_revision_requests(self) -> List[RevisionRequest]:
        reqs = list(self._plan_revision_requests)
        self._plan_revision_requests.clear()
        return reqs

    def run_experiment(
        self,
        experiment: ExperimentSpec,
        alternatives: Optional[List[ExperimentSpec]] = None,
        method_description: str = "",
    ) -> ExperimentOutput:
        """
        Run with PIVOT/REFINE loop.
        alternatives: other designs from Planner for PIVOT.
        method_description is retained for callers/logging but is not used for claim checks.
        """
        log_agent_action("EngineerAgent", "start_experiment", {"experiment": experiment.get("name")})
        contract_hash = experiment.get("contract_hash")
        alternatives = [] if contract_hash else (alternatives or experiment.get("alternatives") or [])
        max_attempts = 4
        approach = experiment
        decision_log = []
        code = ""
        self._progress("experiment_start", {
            "experiment": experiment.get("name"),
            "max_attempts": max_attempts,
            "seeds": self.runtime_config.experiment_seeds,
        })

        for attempt in range(1, max_attempts + 1):
            self._progress("experiment_attempt", {
                "experiment": approach.get("name"),
                "attempt": attempt,
                "max_attempts": max_attempts,
                "action": "generate_code",
            })
            # Prefer previously refined code when REFINE left it on approach
            if approach.get("_refined_code"):
                code = approach.pop("_refined_code")
            else:
                code = self._generate_experiment_code(approach)

            if not (code or "").strip():
                reason = "empty_code_generation"
                decision_log.append({"attempt": attempt, "decision": "REFINE", "reason": reason})
                self._progress("experiment_refine", {
                    "experiment": approach.get("name"),
                    "attempt": attempt,
                    "reason": reason,
                })
                if attempt >= max_attempts:
                    self.request_plan_revision(reason, approach, detail="code generation returned empty string")
                    return self._fail(approach, reason, decision_log, code, failure_kind=reason)
                approach = {
                    **approach,
                    "refine_feedback": (
                        "Previous generation returned NO code (empty string). "
                        "Return COMPLETE runnable Python only — no markdown fences, no prose."
                    ),
                }
                continue

            ok, err = validate_code(code)
            if not ok:
                decision_log.append({"attempt": attempt, "decision": "REFINE", "reason": err})
                self._progress("experiment_refine", {
                    "experiment": approach.get("name"),
                    "attempt": attempt,
                    "reason": err,
                })
                code = self._refine_code(code, err, approach)
                ok, err = validate_code(code)
                if not ok:
                    if attempt >= max_attempts or self._is_plan_level_sandbox_block(err):
                        self.request_plan_revision(
                            "sandbox_blocked_required_api",
                            approach,
                            detail=err,
                        )
                        return self._fail(approach, err, decision_log, code)
                    approach = {**approach, "refine_feedback": err}
                    continue

            known_answer = approach.get("known_answer") or fixture_for(approach) or {}
            if known_answer:
                check = run_known_answer_check(code, known_answer.get("metrics") or {}, float(known_answer.get("tolerance", 1e-3)))
                if not check.get("passed"):
                    detail = json.dumps(check.get("mismatches") or check.get("reason"), default=str)
                    if attempt >= max_attempts:
                        self.request_plan_revision("known_answer_check_failed", approach, detail=detail)
                        return self._fail(
                            approach,
                            "known_answer_check_failed: " + detail,
                            decision_log,
                            code,
                            failure_kind="known_answer_check_failed",
                        )
                    decision_log.append({"attempt": attempt, "decision": "REFINE", "reason": "known_answer_check_failed"})
                    approach = {
                        **approach,
                        "refine_feedback": f"Known-answer check failed: {detail}. Fix the implementation.",
                    }
                    continue

            consistency = self.check_code_claim_consistency(approach, code)
            if not consistency.get("consistent", False):
                detail = "; ".join(
                    consistency.get("notes") or ["generated code does not implement the committed claims"]
                )
                decision_log.append({
                    "attempt": attempt,
                    "decision": "REFINE",
                    "reason": f"code_claim_inconsistency: {detail}",
                })
                self._progress("experiment_refine", {
                    "experiment": approach.get("name"),
                    "attempt": attempt,
                    "reason": "code_claim_inconsistency",
                    "detail": detail[:400],
                })
                if attempt >= max_attempts:
                    self.request_plan_revision("code_claim_inconsistency", approach, detail=detail)
                    return self._fail(
                        approach,
                        "code_claim_inconsistency: " + detail,
                        decision_log,
                        code,
                        failure_kind="code_claim_inconsistency",
                    )
                baselines = approach.get("baselines") or approach.get("baseline_comparison") or []
                components = approach.get("claimed_components") or approach.get("components") or []
                approach = {
                    **approach,
                    "refine_feedback": (
                        f"Your last attempt was inconsistent with the experiment contract: {detail}. "
                        f"Implement the claimed baselines/components exactly, or drop any unimplemented claim. "
                        f"Contract baselines={baselines!r}; claimed_components={components!r}."
                    ),
                }
                continue

            self._progress("experiment_attempt", {
                "experiment": approach.get("name"),
                "attempt": attempt,
                "action": "multi_seed",
                "seeds": self.runtime_config.experiment_seeds,
            })
            multi = run_multi_seed(code, n_seeds=self.runtime_config.experiment_seeds)

            if multi.get("success") and multi.get("aggregate_metrics"):
                self._progress("experiment_ablation", {
                    "experiment": approach.get("name"),
                    "attempt": attempt,
                })
                ablation = self._auto_ablation(approach, code)
                raw_path = self._store_raw(approach["name"], multi)
                output = {
                    "experiment_name": approach["name"],
                    "code": code,
                    "multi_seed": multi,
                    "aggregate_metrics": multi["aggregate_metrics"],
                    "results": {"metrics": {k: v["mean"] for k, v in multi["aggregate_metrics"].items()}},
                    "validation": {"is_valid": True, "issues": [], "warnings": []},
                    "ablation": ablation,
                    "code_claim_consistency": consistency,
                    "success": True,
                    "decision_log": decision_log,
                    "raw_results_path": raw_path,
                    "timestamp": str(datetime.now()),
                    "approach": approach.get("name"),
                    "contract_hash": contract_hash,
                    "outcome": approach.get("hypothesis_outcome") or "inconclusive",
                    "attempts": decision_log,
                }
                self._store_experiment_results(output, approach)
                tracker = get_tracker()
                if tracker:
                    self.ledger.record_event("experiment_artifact", {"experiment": approach["name"], "raw_results_path": raw_path, "seeds": self.runtime_config.experiment_seeds, "metrics": output["aggregate_metrics"]}, tracker.run_id, "EngineerAgent")
                    self.ledger.record_artifact(tracker.run_id, "raw_experiment_results", raw_path, {"experiment": approach["name"], "metrics": output["aggregate_metrics"]})
                self._progress("experiment_complete", {
                    "experiment": approach["name"],
                    "success": True,
                    "attempt": attempt,
                    "metrics": output["aggregate_metrics"],
                    "raw_results_path": raw_path,
                })
                log_agent_action("EngineerAgent", "experiment_complete", {
                    "experiment": approach["name"],
                    "success": True,
                    "attempt": attempt,
                })
                return output

            # Failure path: REFINE or PIVOT
            error = multi.get("error") or "underperformed / no metrics"
            decision = self._decide_pivot_or_refine(approach, error, alternatives, attempt)
            decision_log.append({"attempt": attempt, "decision": decision, "reason": error})
            tracker = get_tracker()
            self._progress("experiment_decision", {
                "experiment": approach.get("name"),
                "attempt": attempt,
                "decision": decision,
                "reason": str(error)[:400],
            })
            if decision == "REFINE":
                if tracker:
                    tracker.bump("refines")
                refined = self._refine_code(code, error, approach)
                approach = {**approach, "refine_feedback": error, "_refined_code": refined}
            else:  # PIVOT
                if tracker:
                    tracker.bump("pivots")
                CrossRunMemory().record_pivot(approach.get("name", "?"), error)
                if alternatives:
                    approach = alternatives.pop(0)
                else:
                    self.request_plan_revision("no_alternatives_after_pivot", approach, detail=error)
                    return self._fail(approach, error, decision_log, code)

        self.request_plan_revision("max_attempts_exhausted", approach, detail="bounded code-only repair attempts exhausted")
        return self._fail(approach, "max_attempts_exhausted", decision_log, code)

    def run_branching_search(
        self,
        contribution_experiments: List[ExperimentSpec],
        method_description: str = "",
    ) -> ExperimentOutput:
        """
        Architecture 8.1: cheap parallel short runs, promote winner to full multi-seed.
        Each item may have 'variants' list; otherwise treat as single design.
        """
        candidates = []
        for exp in contribution_experiments:
            variants = exp.get("variants") or exp.get("alternatives") or [exp]
            for v in variants[: self.runtime_config.experiment_branch_count]:
                candidates.append(v)

        cheap_scores = []
        for cand in candidates:
            code = self._generate_experiment_code({**cand, "cheap_mode": True})
            # Single seed cheap probe
            probe = execute_sandboxed(code, seed=42)
            score = 0.0
            if probe.get("success"):
                metrics = (probe.get("parsed") or {}).get("metrics") or {}
                nums = [float(v) for v in metrics.values() if isinstance(v, (int, float))]
                score = sum(nums) / len(nums) if nums else 0.5
            cheap_scores.append((score, cand, code, probe))
            tracker = get_tracker()
            if tracker:
                tracker.scratch("EngineerAgent", "cheap_probe", {"name": cand.get("name"), "score": score})
            self._progress("cheap_probe", {
                "name": cand.get("name"),
                "score": score,
                "success": bool(probe.get("success")),
                "error": (probe.get("error") or "")[:200],
            })

        if not cheap_scores:
            return {"success": False, "error": "no candidates"}

        cheap_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_cand, _, _ = cheap_scores[0]
        # Full run on winner
        full = self.run_experiment(best_cand, alternatives=[c for _, c, _, _ in cheap_scores[1:]], method_description=method_description)
        full["branch_search"] = {
            "probed": [{"name": c.get("name"), "score": s} for s, c, _, _ in cheap_scores],
            "winner": best_cand.get("name"),
            "winner_probe_score": best_score,
        }
        return full

    def check_code_claim_consistency(self, experiment: ExperimentSpec, code: str) -> CodeClaimReport:
        """Compare code only against planner-authored baselines/claimed_components."""
        notes = []
        score = 10.0
        claim_markers = {
            "xgboost": ["xgboost", "XGB"],
            "random forest": ["RandomForest"],
            "neural network": ["nn.", "torch", "tensorflow", "keras", "MLP", "MLPClassifier"],
            "svm": ["SVC", "SVR", "SVM"],
            "gradient boosting": ["GradientBoosting", "xgboost", "LightGBM", "lgb"],
            "logistic regression": ["LogisticRegression"],
            "ridge": ["Ridge"],
            "transformer": ["Transformer", "Attention", "Bert"],
            "lstm": ["LSTM"],
            "cnn": ["Conv2d", "Conv1d", "CNN"],
        }
        aliases = {
            "random_forest": "random forest",
            "rf": "random forest",
            "neural_network": "neural network",
            "mlp": "neural network",
            "nn": "neural network",
            "support_vector": "svm",
            "support vector": "svm",
            "gradient_boosting": "gradient boosting",
            "gbdt": "gradient boosting",
            "logistic_regression": "logistic regression",
            "logreg": "logistic regression",
        }

        if isinstance(experiment, str):
            # Legacy callers mistakenly passed free-text method prose; ignore it.
            claim_terms: List[str] = []
        else:
            baselines = experiment.get("baselines") or experiment.get("baseline_comparison") or []
            if isinstance(baselines, str):
                baselines = [baselines]
            components = experiment.get("claimed_components") or experiment.get("components") or []
            if isinstance(components, str):
                components = [components]
            claim_terms = [str(item).strip() for item in list(baselines) + list(components) if str(item).strip()]

        code_l = (code or "").lower()
        matched_keys = set()
        for term in claim_terms:
            normalized = term.lower().replace("-", " ").replace("_", " ").strip()
            compact = normalized.replace(" ", "_")
            key = aliases.get(compact) or aliases.get(normalized)
            if not key:
                for claim_key in claim_markers:
                    if claim_key in normalized or normalized in claim_key:
                        key = claim_key
                        break
            if key:
                matched_keys.add(key)

        for claim in matched_keys:
            markers = claim_markers[claim]
            if not any(marker.lower() in code_l for marker in markers):
                notes.append(f"Contract claims '{claim}' but code lacks {markers}")
                score -= 3.0

        # Detect simplified stub
        if "pass  # TODO" in (code or "") or "NotImplemented" in (code or ""):
            notes.append("Code contains stubs/NotImplemented")
            score -= 4.0

        return {
            "consistent": score >= 8.0 and not notes,
            "score": max(0.0, score),
            "notes": notes,
        }

    @staticmethod
    def _is_plan_level_sandbox_block(error: str) -> bool:
        err_l = (error or "").lower()
        return any(
            token in err_l
            for token in (
                "forbidden",
                "blocked",
                "import blocked",
                "sandbox violation",
                "not allowed",
            )
        )

    def _decide_pivot_or_refine(
        self,
        approach: Dict[str, Any],
        error: str,
        alternatives: List[Dict[str, Any]],
        attempt: int,
    ) -> str:
        err_l = (error or "").lower()
        # Bugs → REFINE; conceptual/API failures → PIVOT
        if any(k in err_l for k in ("syntax", "nameerror", "typeerror", "indent", "sandbox rejection")):
            return "REFINE"
        if attempt >= 2 and alternatives:
            return "PIVOT"
        if any(k in err_l for k in ("unsupported", "no module", "api", "parameter", "assumption")):
            return "PIVOT" if alternatives else "REFINE"
        return "REFINE" if attempt < 3 else ("PIVOT" if alternatives else "REFINE")

    def _generate_experiment_code(self, experiment: Dict[str, Any]) -> str:
        cheap = experiment.get("cheap_mode")
        seeds_note = f"Use at least deterministic seeding. Report metrics as JSON on last stdout line."
        prompt = f"""
Generate COMPLETE runnable Python for this experiment.
Allowed imports ONLY: numpy, pandas, matplotlib, sklearn, scipy, math, statistics, random, json, re, collections, seaborn, networkx, sympy.
FORBIDDEN: subprocess, os, sys, socket, requests, pathlib, open() for writing, exit(), eval(), exec().

Experiment: {experiment.get('name')}
Purpose: {experiment.get('purpose')}
Methodology: {experiment.get('methodology')}
Baselines: {experiment.get('baselines') or experiment.get('baseline_comparison')}
Claimed components: {experiment.get('claimed_components') or experiment.get('components') or []}
Metrics: {experiment.get('evaluation_metrics')}
Data: {experiment.get('data_requirements')} (use synthetic data if needed)
Falsifiable prediction: {experiment.get('falsifiable_prediction', 'N/A')}
Statistical test: {experiment.get('statistical_test', 'N/A')}
{'CHEAP MODE: small n_samples (<=200), fast model, no plots.' if cheap else 'Full mode: reasonable sample size.'}
Refine feedback: {experiment.get('refine_feedback', 'none')}

{seeds_note}
Print a single JSON line: {{"metrics": {{...}}, "raw": {{...optional arrays...}}}}
Return ONLY Python code.
"""
        try:
            code = call_llm(prompt, temperature=0.2, tier="cheap")
            return self._clean_code(code)
        except Exception as e:
            log_agent_action("EngineerAgent", "code_generation_error", {"error": str(e)})
            return self._generate_fallback_code(experiment)

    def _refine_code(self, code: str, error: str, experiment: Dict[str, Any]) -> str:
        prompt = f"""
Fix this experiment code. Error:
{error}

Code:
```python
{code}
```

Constraints: no subprocess/os/exit/eval. Print JSON metrics line.
Return ONLY fixed Python code.
"""
        fixed = call_llm(prompt, temperature=0.1, tier="cheap")
        return self._clean_code(fixed) if fixed else code

    def _auto_ablation(self, experiment: Dict[str, Any], full_code: str) -> Dict[str, Any]:
        """Generate a minimal ablation: disable one claimed component."""
        components = experiment.get("components") or experiment.get("claimed_components") or []
        if not components:
            # Heuristic: look for feature engineering block comments
            components = ["main_component"]
        abl_results = {}
        for comp in components[:2]:
            prompt = f"""
Modify this code to ABLATE (remove/disable) component '{comp}' while keeping the rest.
Original:
```python
{full_code[:4000]}
```
Return ONLY Python. Still print JSON metrics.
"""
            abl_code = self._clean_code(call_llm(prompt, temperature=0.2, tier="cheap") or "")
            if not abl_code:
                continue
            ok, _ = validate_code(abl_code)
            if not ok:
                continue
            run = execute_sandboxed(abl_code, seed=42)
            abl_results[comp] = {
                "success": run.get("success"),
                "metrics": (run.get("parsed") or {}).get("metrics"),
            }
        return abl_results

    def _store_raw(self, name: str, multi: Dict[str, Any]) -> str:
        safe = re.sub(r"[^\w\-]+", "_", name)[:80]
        path = os.path.join(self.raw_dir, f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(multi, f, indent=2, default=str)
        return path

    def _fail(self, experiment, error, decision_log, code="", failure_kind: str = "technical"):
        # Give-up guard: require concrete artifact
        artifact = {"error": str(error), "decision_log": decision_log, "code_snippet": (code or "")[:500]}
        if not artifact["error"] or artifact["error"] == "this isn't feasible":
            artifact["suspicious_bare_claim"] = True
        error_text = str(error)
        if failure_kind == "technical":
            if error_text.startswith("empty_code_generation"):
                failure_kind = "empty_code_generation"
            elif error_text.startswith("code_claim_inconsistency"):
                failure_kind = "code_claim_inconsistency"
            elif error_text.startswith("known_answer_check_failed"):
                failure_kind = "known_answer_check_failed"
            elif error_text == "max_attempts_exhausted":
                failure_kind = "max_attempts_exhausted"
        out = {
            "experiment_name": experiment.get("name"),
            "error": error_text,
            "success": False,
            "decision_log": decision_log,
            "failure_artifact": artifact,
            "code": (code or "")[:2000],
            "timestamp": str(datetime.now()),
            "contract_hash": experiment.get("contract_hash"),
            "failure_kind": failure_kind,
            "attempts": decision_log,
        }
        self._progress("experiment_failed", {
            "experiment": experiment.get("name"),
            "error": str(error)[:400],
            "decision_log": decision_log,
        })
        # Persist failure so Experiments tab / Meta summary can show it
        try:
            name = experiment.get("name") or "failed_experiment"
            path = os.path.join(self.raw_dir, f"FAIL_{re.sub(r'[^\\w\\-]+', '_', name)[:60]}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, default=str)
            out["raw_results_path"] = path
            tracker = get_tracker()
            if tracker:
                self.ledger.record_artifact(tracker.run_id, "failed_experiment", path, {"experiment": name, "error": str(error)})
        except Exception:
            pass
        return out

    def _clean_code(self, code: str) -> str:
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    def _generate_fallback_code(self, experiment: Dict[str, Any]) -> str:
        return f'''
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

np.random.seed(42)
n_samples = 400
X = np.random.randn(n_samples, 8)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

baseline = LogisticRegression(max_iter=200)
baseline.fit(X_train, y_train)
base_pred = baseline.predict(X_test)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

metrics = {{
    "accuracy": float(accuracy_score(y_test, pred)),
    "f1": float(f1_score(y_test, pred)),
    "baseline_accuracy": float(accuracy_score(y_test, base_pred)),
}}
print(json.dumps({{"metrics": metrics, "raw": {{"y_test": y_test.tolist(), "pred": pred.tolist()}}}}))
'''

    def _store_experiment_results(self, output: Dict[str, Any], experiment: Dict[str, Any]):
        try:
            results_file = os.path.join(self.output_dir, f"{experiment['name']}_results.json")
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)
            self.vector_memory.add_embedding(
                generate_embedding(json.dumps({"name": experiment["name"], "metrics": output.get("aggregate_metrics")})),
                {
                    "type": "experiment_results",
                    "namespace": "engineer",
                    "content_class": "generated_narrative",
                    "retrieval_eligible": False,
                    "agent": "EngineerAgent",
                    "outcome_status": "released" if output.get("success") else "failed",
                    "experiment": experiment["name"],
                    "success": output["success"],
                    "results_file": results_file,
                    "timestamp": output["timestamp"],
                },
            )
            tracker = get_tracker()
            if tracker:
                tracker.scratch("EngineerAgent", "results", output.get("aggregate_metrics"))
        except Exception as e:
            log_agent_action("EngineerAgent", "storage_error", {"error": str(e)})
