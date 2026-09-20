"""
EngineerAgent — verifiable, self-healing experiment runner.
Sandbox lockdown, multi-seed, PIVOT/REFINE, ablations, code-claim checks.
"""

from __future__ import annotations

import json
import logging
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
from core.sandbox import run_known_answer_check, validate_code
from core.sandbox_dispatch import execute as sandbox_execute, execute_multi_seed
from core.run_log import get_tracker, CrossRunMemory, emit_event
from core.research_db import research_db
from core.contracts import CodeClaimReport, ExperimentOutput, ExperimentSpec, RevisionRequest
from core.known_answers import fixture_for
from core.datasets import download_hf_dataset, get_dataset_info, load_local_dataset

logger = logging.getLogger(__name__)


def _effectively_empty_code(code: str) -> bool:
    """Return True if code contains only comments/docstrings with no runnable statements."""
    code_text = (code or "").strip()
    if not code_text:
        return True
    try:
        import ast
        tree = ast.parse(code_text)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Assign, ast.For, ast.While,
                                 ast.If, ast.With, ast.FunctionDef, ast.ClassDef,
                                 ast.Return, ast.Import, ast.ImportFrom)):
                return False
            if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
                return False
        return True
    except SyntaxError:
        return not bool(code_text)


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

    def prepare_dataset(self, dataset_plan: str, max_samples: int = 1000, dataset_resolutions: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """Prepare a dataset for experiment execution.

        Checks Planner's pre-resolved info first, then local catalog, then downloads from HuggingFace.
        Returns dataset dict with 'rows', 'row_count', 'features' or None.
        """
        if not dataset_plan or not dataset_plan.strip():
            logger.info("prepare_dataset: no dataset plan provided, skipping")
            return None
        plan = dataset_plan.strip()
        logger.info(f"prepare_dataset: preparing dataset '{plan}' (max_samples={max_samples})")

        # Check if Planner already resolved this dataset
        if dataset_resolutions:
            for res in dataset_resolutions:
                if isinstance(res, dict) and res.get("dataset") == plan:
                    if res.get("status") == "local_loaded":
                        # Already loaded by Planner — reload from local
                        try:
                            local = load_local_dataset(plan)
                            if local and local.get("row_count", 0) > 0:
                                logger.info(f"prepare_dataset: loaded planner-resolved local dataset '{plan}', rows={local['row_count']}")
                                return local
                        except Exception as e:
                            logger.debug("Local dataset reload failed: %s", e)
                    # Info was resolved — proceed to download

        # Try local catalog
        try:
            local = load_local_dataset(plan)
            if local and local.get("row_count", 0) > 0:
                logger.info(f"prepare_dataset: loaded from local catalog '{plan}', rows={local['row_count']}, source=local")
                log_agent_action("Engineer", "dataset_loaded_local", {
                    "name": plan, "rows": local["row_count"],
                })
                return local
        except Exception as e:
            logger.debug("Local catalog load failed: %s", e)
        # Try HuggingFace download
        logger.info(f"prepare_dataset: local catalog miss for '{plan}', attempting HuggingFace download")
        hf_result = download_hf_dataset(plan, max_samples=max_samples)
        if hf_result:
            logger.info(f"prepare_dataset: downloaded from HuggingFace '{plan}', rows={hf_result.get('row_count', 0)}, source=HuggingFace")
            log_agent_action("Engineer", "dataset_downloaded_hf", {
                "name": plan, "rows": hf_result.get("row_count", 0),
            })
            return hf_result
        # Fallback: get info only (no download)
        info = get_dataset_info(plan)
        if info:
            logger.info(f"prepare_dataset: no data obtained for '{plan}', info-only source={info.get('source')}")
            log_agent_action("Engineer", "dataset_info_only", {
                "name": plan, "source": info.get("source"),
            })
        else:
            logger.warning(f"prepare_dataset: no dataset found for '{plan}' (local, HF, or info all failed)")
        return None

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
        exp_name = experiment.get("name", "<unnamed>")
        logger.info(f"run_experiment: starting experiment '{exp_name}'")
        logger.info(f"run_experiment: config — seeds={self.runtime_config.experiment_seeds}, max_attempts=4, contract_hash={experiment.get('contract_hash', 'none')}")
        log_agent_action("EngineerAgent", "start_experiment", {"experiment": exp_name})
        contract_hash = experiment.get("contract_hash")
        alternatives = [] if contract_hash else (alternatives or experiment.get("alternatives") or [])
        max_attempts = 4
        approach = experiment
        original_experiment_name = experiment.get("name")
        decision_log = []
        code = ""
        last_error = ""
        last_failure_kind = "technical"
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
            if approach.get("_refined_code"):
                code = approach.pop("_refined_code")
                logger.info(f"run_experiment: [{exp_name}] attempt {attempt}/{max_attempts} — using pre-refined code ({len(code)} chars)")
            else:
                logger.info(f"run_experiment: [{exp_name}] attempt {attempt}/{max_attempts} — generating code via LLM")
                code = self._generate_experiment_code(approach)
                logger.info(f"run_experiment: [{exp_name}] code generated — {len(code)} chars")

            if not (code or "").strip():
                reason = "empty_code_generation"
                logger.info(f"run_experiment: [{exp_name}] attempt {attempt} — code generation returned empty, requesting REFINE")
                decision_log.append({"attempt": attempt, "decision": "REFINE", "reason": reason})
                self._progress("experiment_refine", {
                    "experiment": approach.get("name"),
                    "attempt": attempt,
                    "reason": reason,
                })
                last_error = reason
                last_failure_kind = reason
                if attempt >= max_attempts:
                    self.request_plan_revision(reason, approach, detail="code generation returned empty string")
                approach = {
                    **approach,
                    "refine_feedback": (
                        "Previous generation returned NO code (empty string). "
                        "Return COMPLETE runnable Python only — no markdown fences, no prose."
                    ),
                }
                continue

            if (code or "").strip() and _effectively_empty_code(code):
                reason = "empty_code_generation"
                logger.info(f"run_experiment: [{exp_name}] attempt {attempt} — code contains only comments/docstrings, requesting REFINE")
                decision_log.append({"attempt": attempt, "decision": "REFINE", "reason": reason})
                self._progress("experiment_refine", {
                    "experiment": approach.get("name"),
                    "attempt": attempt,
                    "reason": reason,
                })
                last_error = reason
                last_failure_kind = reason
                if attempt >= max_attempts:
                    self.request_plan_revision(reason, approach, detail="code contains only comments/docstrings")
                approach = {
                    **approach,
                    "refine_feedback": (
                        "Previous generation returned only comments or docstrings with no runnable code. "
                        "Return COMPLETE runnable Python with actual computation — no markdown fences, no prose."
                    ),
                }
                continue

            ok, err = validate_code(code)
            if not ok:
                logger.info(f"run_experiment: [{exp_name}] attempt {attempt} — sandbox validation failed: {err[:200]}")
                decision_log.append({"attempt": attempt, "decision": "REFINE", "reason": err})
                self._progress("experiment_refine", {
                    "experiment": approach.get("name"),
                    "attempt": attempt,
                    "reason": err,
                })
                code = self._refine_code(code, err, approach)
                ok, err = validate_code(code)
                if not ok:
                    if self._is_plan_level_sandbox_block(err):
                        self.request_plan_revision("sandbox_blocked_required_api", approach, detail=err)
                        return self._fail(approach, err, decision_log, code, original_name=original_experiment_name)
                    last_error = err
                    last_failure_kind = "technical"
                    if attempt >= max_attempts:
                        self.request_plan_revision("sandbox_blocked_required_api", approach, detail=err)
                    approach = {**approach, "refine_feedback": err}
                    continue

            known_answer = approach.get("known_answer") or fixture_for(approach) or {}
            if known_answer:
                logger.info(f"run_experiment: [{exp_name}] attempt {attempt} — running known-answer check")
                check = run_known_answer_check(code, known_answer.get("metrics") or {}, float(known_answer.get("tolerance", 1e-3)))
                if not check.get("passed"):
                    detail = json.dumps(check.get("mismatches") or check.get("reason"), default=str)
                    decision_log.append({"attempt": attempt, "decision": "REFINE", "reason": "known_answer_check_failed"})
                    last_error = "known_answer_check_failed: " + detail
                    last_failure_kind = "known_answer_check_failed"
                    if attempt >= max_attempts:
                        self.request_plan_revision("known_answer_check_failed", approach, detail=detail)
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
                logger.info(f"run_experiment: [{exp_name}] attempt {attempt} — code-claim inconsistency: {detail[:300]}")
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
                last_error = "code_claim_inconsistency: " + detail
                last_failure_kind = "code_claim_inconsistency"
                if attempt >= max_attempts:
                    self.request_plan_revision("code_claim_inconsistency", approach, detail=detail)
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
            logger.info(f"run_experiment: [{exp_name}] attempt {attempt} — executing multi-seed run (n_seeds={self.runtime_config.experiment_seeds})")
            multi = execute_multi_seed(code, n_seeds=self.runtime_config.experiment_seeds)

            if multi.get("success") and multi.get("aggregate_metrics"):
                logger.info(f"run_experiment: [{exp_name}] attempt {attempt} — execution succeeded, extracting metrics: {list(multi['aggregate_metrics'].keys())}")
                self._progress("experiment_ablation", {
                    "experiment": approach.get("name"),
                    "attempt": attempt,
                })
                ablation = self._auto_ablation(approach, code)
                raw_path = self._store_raw(approach["name"], multi)
                output = {
                    "experiment_name": approach["name"],
                    "original_experiment_name": original_experiment_name,
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
                logger.info(f"run_experiment: [{exp_name}] attempt {attempt} — experiment COMPLETE, metrics={output['aggregate_metrics']}, raw={raw_path}")
                log_agent_action("EngineerAgent", "experiment_complete", {
                    "experiment": approach["name"],
                    "success": True,
                    "attempt": attempt,
                })
                return output

            # Failure path: REFINE or PIVOT
            error = multi.get("error") or "underperformed / no metrics"
            decision = self._decide_pivot_or_refine(approach, error, alternatives, attempt)
            logger.info(f"run_experiment: [{exp_name}] attempt {attempt} — execution failed ({error[:200]}), decision={decision}")
            decision_log.append({"attempt": attempt, "decision": decision, "reason": error})
            tracker = get_tracker()
            self._progress("experiment_decision", {
                "experiment": approach.get("name"),
                "attempt": attempt,
                "decision": decision,
                "reason": str(error)[:400],
            })
            last_error = error
            last_failure_kind = "technical"
            if decision == "REFINE":
                if tracker:
                    tracker.bump("refines")
                local_ctx = multi.get("local_code_context", "")
                refined = self._refine_code(code, error, approach, local_context=local_ctx)
                approach = {**approach, "refine_feedback": error, "_refined_code": refined}
            else:  # PIVOT
                if tracker:
                    tracker.bump("pivots")
                CrossRunMemory().record_pivot(approach.get("name", "?"), error)
                if alternatives:
                    approach = alternatives.pop(0)
                else:
                    self.request_plan_revision("no_alternatives_after_pivot", approach, detail=error)
                    return self._fail(approach, error, decision_log, code, original_name=original_experiment_name)

        logger.warning(f"run_experiment: [{exp_name}] max_attempts={max_attempts} exhausted, failing experiment")
        self.request_plan_revision("max_attempts_exhausted", approach, detail="bounded code-only repair attempts exhausted")
        return self._fail(approach, last_error or "max_attempts_exhausted", decision_log, code, failure_kind=last_failure_kind, original_name=original_experiment_name)

    def run_branching_search(
        self,
        contribution_experiments: List[ExperimentSpec],
        method_description: str = "",
    ) -> ExperimentOutput:
        """
        Architecture 8.1: cheap parallel short runs, promote winner to full multi-seed.
        Each item may have 'variants' list; otherwise treat as single design.
        """
        logger.info(f"run_branching_search: starting branching search with {len(contribution_experiments)} experiment(s)")
        candidates = []
        for exp in contribution_experiments:
            variants = exp.get("variants") or exp.get("alternatives") or [exp]
            for v in variants[: self.runtime_config.experiment_branch_count]:
                candidates.append(v)
        logger.info(f"run_branching_search: {len(candidates)} candidate variant(s) to probe (branch_count={self.runtime_config.experiment_branch_count})")

        cheap_scores = []
        for cand in candidates:
            cand_name = cand.get("name", "<unnamed>")
            logger.info(f"run_branching_search: probing variant '{cand_name}'")
            code = self._generate_experiment_code({**cand, "cheap_mode": True})
            # Single seed cheap probe
            probe = sandbox_execute(code, seed=42)
            score = 0.0
            probe_ok = False
            if probe.get("success"):
                parsed = probe.get("parsed") or {}
                metrics = parsed.get("metrics") or {}
                nums = [float(v) for v in metrics.values() if isinstance(v, (int, float))]
                if nums:
                    score = sum(nums) / len(nums)
                    probe_ok = True
                else:
                    # Probe ran but produced no parseable metrics — not a pass
                    score = 0.0
                    probe_ok = False
            cheap_scores.append((score, cand, code, probe, probe_ok))
            logger.info(f"run_branching_search: variant '{cand_name}' — score={score:.4f}, probe_ok={probe_ok}")
            tracker = get_tracker()
            if tracker:
                tracker.scratch("EngineerAgent", "cheap_probe", {"name": cand.get("name"), "score": score, "probe_ok": probe_ok})
            self._progress("cheap_probe", {
                "name": cand.get("name"),
                "score": score,
                "success": bool(probe.get("success")),
                "probe_ok": probe_ok,
                "error": (probe.get("error") or "")[:200],
            })

        if not cheap_scores:
            logger.warning("run_branching_search: no candidates to probe, returning failure")
            return {"success": False, "error": "no candidates"}

        # Sort by score descending, but only consider probes that actually
        # produced parseable metrics.  Probes that errored or produced no
        # metrics get score 0 and are deprioritized — not promoted.
        cheap_scores.sort(key=lambda x: (x[4], x[0]), reverse=True)
        best_score, best_cand, _, _, _ = cheap_scores[0]
        logger.info(f"run_branching_search: winner selected — '{best_cand.get('name', '<unnamed>')}' (probe_score={best_score:.4f})")
        # Full run on winner, passing remaining candidates as alternatives for PIVOT
        full = self.run_experiment(
            best_cand,
            alternatives=[c for _, c, _, _, _ in cheap_scores[1:]],
            method_description=method_description,
        )
        full["branch_search"] = {
            "probed": [{"name": c.get("name"), "score": s, "probe_ok": ok} for s, c, _, _, ok in cheap_scores],
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

        # --- Computation verification: detect hardcoded-literal fabricated metrics ---
        comp_note = self._check_code_has_computation(code)
        if comp_note:
            notes.append(comp_note)
            score -= 5.0

        return {
            "consistent": score >= 8.0 and not notes,
            "score": max(0.0, score),
            "notes": notes,
        }

    @staticmethod
    def _check_code_has_computation(code: str) -> Optional[str]:
        """Return a note if code appears to contain no real computation before its
        final ``print(json.dumps(...))`` line, else ``None``.

        Heuristic: parse the AST and look for evidence that the code actually
        trains a model, runs a numeric pipeline, or calls into sklearn/scipy/numpy
        *before* producing its output.  Hardcoded literal dicts bypass this gate
        silently today — this method catches that class of fabrication.
        """
        import ast as _ast

        if not (code or "").strip():
            return None

        try:
            tree = _ast.parse(code)
        except SyntaxError:
            return None  # let the normal syntax-check path handle this

        # Computation evidence markers — calls that imply actual work
        _FIT_NAMES = {"fit", "fit_transform", "fit_predict", "partial_fit",
                       "train_test_split", "cross_val_score", "GridSearchCV",
                       "RandomizedSearchCV", "Pipeline", "StandardScaler",
                       "MinMaxScaler", "PCA", "select_kbest", "kmeans",
                       "curve_fit", "minimize", "fsolve", "odeint"}
        _NUMPY_CALLS = {"array", "linspace", "arange", "random", "randn",
                         "zeros", "ones", "diag", "dot", "matmul", "einsum",
                         "mean", "std", "var", "sum", "cumsum", "diff",
                         "histogram", "percentile", "argsort", "where"}
        _COMPUTATION_MODULES = {"sklearn", "scipy", "statsmodels", "xgboost",
                                 "lightgbm", "torch", "tensorflow", "keras"}

        has_fit_call = False
        has_numpy_call = False
        has_computation_import = False
        has_numeric_assignment = False

        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in _COMPUTATION_MODULES:
                        has_computation_import = True
            elif isinstance(node, _ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in _COMPUTATION_MODULES:
                        has_computation_import = True
            elif isinstance(node, _ast.Call):
                func_name = ""
                if isinstance(node.func, _ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, _ast.Name):
                    func_name = node.func.id
                if func_name in _FIT_NAMES:
                    has_fit_call = True
                if func_name in _NUMPY_CALLS:
                    has_numpy_call = True
            # Detect numeric binary ops on variables (e.g. np.mean(...), scores.mean())
            elif isinstance(node, _ast.BinOp) and isinstance(node.op, (_ast.Add, _ast.Sub, _ast.Mult, _ast.Div)):
                has_numeric_assignment = True

        if not (has_fit_call or has_numpy_call or has_computation_import or has_numeric_assignment):
            return "Code contains no evidence of computation (no model fitting, numpy calls, or numeric operations)"

        # Second check: the final print(json.dumps(...)) — are the metrics all literal constants?
        # Walk the AST looking for the last Expr(Call(print, ...)) and inspect its arguments.
        print_calls = [n for n in _ast.walk(tree)
                       if isinstance(n, _ast.Expr) and isinstance(n.value, _ast.Call)
                       and isinstance(n.value.func, _ast.Name) and n.value.func.id == "print"]
        if not print_calls:
            return None  # no final print — can't检验

        last_print = print_calls[-1]
        # Check if the argument to print is a dict with all-numeric-literal values
        arg = last_print.value.args[0] if last_print.value.args else None
        if arg is None:
            return None

        def _all_literals_in_dict(node: _ast.AST) -> bool:
            """True if node is a Dict whose values are all numeric/string literals."""
            if not isinstance(node, _ast.Dict):
                return False
            for val in node.values:
                if isinstance(val, _ast.Constant) and isinstance(val.value, (int, float, str, bool)):
                    continue
                if isinstance(val, _ast.Dict):
                    if not _all_literals_in_dict(val):
                        return False
                elif isinstance(val, _ast.List):
                    if not all(
                        isinstance(e, _ast.Constant) and isinstance(e.value, (int, float, str, bool))
                        for e in val.elts
                    ):
                        return False
                else:
                    return False
            return True

        # Unwrap json.dumps(...)
        if isinstance(arg, _ast.Call) and isinstance(arg.func, _ast.Attribute):
            if arg.func.attr == "dumps" and arg.args:
                arg = arg.args[0]

        if _all_literals_in_dict(arg):
            return "Code outputs hardcoded literal metrics with no computation — likely fabricated"

        return None

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
        # No metrics means code ran but produced no useful output - pivot, not refine
        if "no parseable metrics" in err_l or "no metrics" in err_l:
            if attempt >= 2 and alternatives:
                return "PIVOT"
            if attempt >= 3:
                return "PIVOT" if alternatives else "REFINE"
            return "REFINE" if attempt < 3 else ("PIVOT" if alternatives else "REFINE")
        # Bugs → REFINE; conceptual/API failures → PIVOT
        if any(k in err_l for k in ("syntax", "nameerror", "typeerror", "indent", "sandbox rejection")):
            return "REFINE"
        if attempt >= 2 and alternatives:
            return "PIVOT"
        if any(k in err_l for k in ("unsupported", "no module", "api", "parameter", "assumption")):
            return "PIVOT" if alternatives else "REFINE"
        return "REFINE" if attempt < 3 else ("PIVOT" if alternatives else "REFINE")

    def _error_category_hint(error: str) -> str:
        """Deterministic hint for the most common execution failure categories
        (Critic-Experience-Bank style: recurring failures become reusable guidance)."""
        lowered = str(error or "").lower()
        if any(token in lowered for token in ("timeout", "timed out", "too long")):
            return (
                "The previous run exceeded the time budget. Shrink the workload: "
                "smaller n_samples (<=200), fewer iterations, no heavy plots."
            )
        if any(token in lowered for token in ("memory", "memoryerror", "allocation")):
            return "Reduce memory usage: smaller arrays, avoid storing full result matrices."
        if any(token in lowered for token in ("import", "module", "nameerror")):
            return (
                "Only these imports are allowed: numpy, pandas, matplotlib, sklearn, "
                "scipy, math, statistics, random, json, re, collections, seaborn, networkx, sympy."
            )
        if any(token in lowered for token in ("forbidden", "sandbox", "blocked", "open(")):
            return "The sandbox forbids file/subprocess access. Keep all data synthetic and in-memory."
        if "json" in lowered or "metrics" in lowered:
            return "The final stdout line must be a single JSON object: {\"metrics\": {...}, \"raw\": {...}}."
        return ""

    def _generate_experiment_code(self, experiment: Dict[str, Any]) -> str:
        cheap = experiment.get("cheap_mode")
        seeds_note = f"Use at least deterministic seeding. Report metrics as JSON on last stdout line."
        try:
            tags = CrossRunMemory().get_prompt_context(limit=6)
        except Exception:
            tags = []
        lessons_lines = []
        for tag in tags:
            label = tag.get("item") or tag.get("experiment") or ""
            category = tag.get("rejection_reason") or tag.get("failure_category") or tag.get("revision_reason") or "unknown"
            if label:
                lessons_lines.append(f"- {tag.get('kind', tag.get('category', 'issue'))} {label}: {category}")
        lessons_block = (
            "\nKnown failure patterns from prior runs (avoid repeating them):\n"
            + "\n".join(lessons_lines)[:900] + "\n"
            if lessons_lines
            else ""
        )
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
{lessons_block}
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

    def _refine_code(self, code: str, error: str, experiment: Dict[str, Any], local_context: str = "") -> str:
        exp_name = experiment.get("name", "<unnamed>")
        hint = self._error_category_hint(error)
        logger.info(f"_refine_code: [{exp_name}] starting refinement, error_category_hint='{hint[:100]}'")
        context_block = ""
        if local_context:
            context_block = f"\nDebugging context (traceback + implicated custom functions):\n{local_context}\n"
        hint_block = f"\nFix hint for this failure category:\n{hint}\n" if hint else ""
        prompt = f"""
Fix this experiment code. Error:
{error}
{context_block}
Code:
```python
{code}
```

Constraints: no subprocess/os/exit/eval. Print JSON metrics line.
{hint_block}
Return ONLY fixed Python code.
"""
        fixed = call_llm(prompt, temperature=0.1, tier="cheap")
        if fixed:
            logger.info(f"_refine_code: [{exp_name}] refinement complete — {len(fixed)} chars")
        else:
            logger.warning(f"_refine_code: [{exp_name}] LLM returned empty refinement, keeping original code")
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
            run = sandbox_execute(abl_code, seed=42)
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

    def _fail(self, experiment, error, decision_log, code="", failure_kind: str = "technical", original_name: str = None):
        # Give-up guard: require concrete artifact
        artifact = {"error": str(error), "decision_log": decision_log, "code_snippet": (code or "")[:500]}
        if not artifact["error"] or artifact["error"] == "this isn't feasible":
            artifact["suspicious_bare_claim"] = True
        error_text = str(error)
        if _effectively_empty_code(code or ""):
            failure_kind = "empty_code_generation"
        elif failure_kind == "technical":
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
            "original_experiment_name": original_name or experiment.get("name"),
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
