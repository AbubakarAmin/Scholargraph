"""
PlannerAgent — falsifiable, revisable experiment designer.
Bidirectional Engineer → Planner revision path; baselines required.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import log_agent_action, parse_json_from_llm
from core.llm import call_llm, generate_embedding
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.contracts import Contribution, Dependency, ExperimentSpec, Plan, RevisionRequest, Timeline, Topic
from core.evidence_gate import validate_experiments
from core.memory import memory
from core.run_log import get_tracker, CrossRunMemory
from core.capabilities import SANDBOX_CAPABILITY_MANIFEST, check_plan_feasibility
from core.datasets import list_datasets


class PlannerAgent:
    """Creates falsifiable research plans with baselines and experiment branches."""

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()

    @property
    def runtime_config(self):
        return self.context.config if self.context else config

    @property
    def vector_memory(self):
        return self.context.memory if self.context else memory

    def create_plan(self, topic: Topic) -> Plan:
        log_agent_action("PlannerAgent", "start_planning", {"topic": topic.get("title")})
        lessons = json.dumps(CrossRunMemory().get_prompt_context(), sort_keys=True)

        plan = self._generate_plan_structure(topic, lessons)
        plan["dataset_catalog"] = list_datasets()
        plan["contributions"] = self._ensure_falsifiable_contributions(plan, topic)
        plan["experiments"] = self._generate_experiments(topic, plan)
        plan["experiments"] = self._normalize_experiments(plan["experiments"])

        unfalsifiable = self._flag_unfalsifiable(plan)
        missing_baselines = self._flag_missing_baselines(plan)
        if unfalsifiable or missing_baselines:
            plan = self._repair_plan(plan, topic, unfalsifiable, missing_baselines)
            plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])

        plan["experiments"] = self._attach_variants(plan.get("experiments") or [])
        plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])

        plan["dependencies"] = self._generate_dependencies(plan)
        plan["timeline"] = self._generate_timeline(plan)
        plan["unfalsifiable_flags"] = self._flag_unfalsifiable(plan)
        plan["missing_baseline_flags"] = self._flag_missing_baselines(plan)
        plan["revision_history"] = []

        schema_errors = validate_experiments(plan.get("experiments"))
        if schema_errors:
            for _ in range(2):
                plan = self._repair_plan(
                    plan,
                    topic,
                    unfalsifiable=[],
                    missing_baselines=[],
                    schema_errors=schema_errors,
                )
                plan["experiments"] = self._attach_variants(plan.get("experiments") or [])
                plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])
                schema_errors = validate_experiments(plan.get("experiments"))
                if not schema_errors:
                    break
            if schema_errors:
                raise ValueError("Plan experiments failed contract validation: " + "; ".join(schema_errors))

        feasibility_errors = check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST)
        if feasibility_errors:
            self._apply_capability_rescope(plan, topic, feasibility_errors)
            plan["methodology"] = str(plan.get("methodology", "")).replace("GPU", "CPU-compatible").replace("download", "use bundled")
            plan["compute_budget"] = "CPU-only, bounded synthetic or bundled small dataset"
            for experiment in plan.get("experiments") or []:
                if isinstance(experiment, dict):
                    dataset = experiment.setdefault("dataset", {})
                    if any(token in str(dataset).lower() for token in ("download", "yahoo", "uci", "internet")):
                        experiment["dataset"] = {"name": "bundled_synthetic", "access_policy": "local-only"}
                    experiment.pop("gpu", None)
        plan["capability_manifest"] = SANDBOX_CAPABILITY_MANIFEST.as_dict()
        executable_plan = {
            "methodology": plan.get("methodology", ""),
            "compute_budget": plan.get("compute_budget", ""),
            "experiments": plan.get("experiments") or [],
        }
        plan["feasibility_errors"] = check_plan_feasibility(executable_plan, SANDBOX_CAPABILITY_MANIFEST)
        if plan["feasibility_errors"]:
            raise ValueError("Plan rejected by sandbox capability manifest: " + "; ".join(plan["feasibility_errors"]))

        self._store_plan(plan, topic)
        log_agent_action("PlannerAgent", "plan_created", {
            "sections": len(plan.get("sections", [])),
            "experiments": len(plan.get("experiments", [])),
            "unfalsifiable": len(plan["unfalsifiable_flags"]),
        })
        return plan

    @staticmethod
    def _apply_capability_rescope(plan: Plan, topic: Topic, reasons: List[str]) -> None:
        """Rewrite publication framing when execution scope changes."""
        original_title = str(plan.get("title") or topic.get("title") or "Research study")
        original_contributions = plan.get("contributions") or plan.get("expected_contributions") or []
        replacement = "bounded local synthetic or bundled benchmark"
        title = original_title
        for token in ("Yahoo Finance", "UCI", "internet", "production", "real-world"):
            title = title.replace(token, replacement)
        if title == original_title:
            title = f"{original_title} on a {replacement}"

        rewritten = []
        for contribution in original_contributions:
            if isinstance(contribution, dict):
                item = dict(contribution)
                if item.get("claim"):
                    item["claim"] = f"{item['claim']} within a {replacement}"
                rewritten.append(item)
            else:
                rewritten.append(f"{contribution} within a {replacement}")

        plan["capability_rescope"] = {
            "from": reasons,
            "original_title": original_title,
            "to": replacement,
            "replacement_dataset_policy": "catalogued or generated locally; no outbound downloads",
        }
        plan["title"] = title
        plan["abstract"] = (
            f"This study evaluates the proposed approach on a {replacement}. "
            "Claims are limited to the recorded local protocol and do not establish performance on the originally requested external setting."
        )
        plan["research_questions"] = [
            f"Under the bounded local protocol, does the proposed method satisfy its falsifiable prediction?"
        ]
        plan["expected_contributions"] = rewritten
        plan["contributions"] = rewritten
        if isinstance(topic, dict):
            topic["title"] = title
            topic["description"] = (
                f"Evaluate the proposed method using a {replacement}; do not claim external-domain or production validity."
            )

    def revise_plan(
        self,
        plan: Plan,
        revision_request: RevisionRequest,
        topic: Optional[Topic] = None,
    ) -> Plan:
        """Bidirectional edge: Engineer requested a plan revision."""
        log_agent_action("PlannerAgent", "revise_plan", revision_request)
        tracker = get_tracker()
        if tracker:
            tracker.bump("plan_revisions")
            tracker.scratch("PlannerAgent", "revision", revision_request)

        prompt = f"""
Revise this research plan based on an Engineer failure or validation request.

Reason: {revision_request.get('reason')}
Detail: {revision_request.get('detail')}
Failed experiment: {revision_request.get('experiment')}

Current plan JSON:
{json.dumps({k: plan.get(k) for k in ('title', 'methodology', 'experiments', 'contributions', 'expected_contributions')}, indent=2)[:6000]}

Requirements:
- Fix unsupported assumptions / APIs / data requirements
- Every contribution needs falsifiable_prediction + statistical_test
- Every experiment MUST be an object with "name", "baselines", "evaluation_metrics", "falsifiable_prediction", "statistical_test"
- Every item in "variants" MUST be a JSON object with at least "name" and "methodology" (e.g. [{{"name": "Variant Name", "methodology": "..."}}]), NOT strings.
Return full updated plan fragment as JSON with keys: methodology, contributions, experiments, revision_notes
"""
        response = call_llm(prompt, temperature=0.4, tier="strong")
        schema_errors: List[str] = []
        for attempt in range(3):
            parsed = parse_json_from_llm(response) or {}
            if isinstance(parsed, dict):
                for key in ("methodology", "contributions", "experiments"):
                    if key in parsed:
                        plan[key] = parsed[key]
                if "experiments" in parsed:
                    plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])
                    plan["experiments"] = self._attach_variants(plan.get("experiments") or [])
                    plan["experiments"] = self._normalize_experiments(plan["experiments"])
                schema_errors = validate_experiments(plan.get("experiments"))
                if not schema_errors:
                    break
                retry_prompt = f"""
Previous revision had schema errors: {'; '.join(schema_errors)}
Please fix the schema errors.
Every experiment MUST be an object with "name", "baselines", "evaluation_metrics", "falsifiable_prediction", "statistical_test".
Every variant in "variants" MUST be an object with "name" and "methodology" (e.g. [{{"name": "Variant A", "methodology": "..."}}]).

Original revision request: {revision_request.get('detail')}
Return JSON with keys: methodology, contributions, experiments, revision_notes
"""
                response = call_llm(retry_prompt, temperature=0.3, tier="strong")
        if schema_errors:
            raise ValueError("Revised plan failed contract validation: " + "; ".join(schema_errors))

        feasibility_errors = check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST)
        if feasibility_errors:
            raise ValueError("Revised plan rejected by sandbox capability manifest: " + "; ".join(feasibility_errors))
        plan.setdefault("revision_history", []).append({
            "request": revision_request,
            "notes": (parse_json_from_llm(response) or {}).get("revision_notes", ""),
            "at": datetime.now().isoformat(),
        })
        CrossRunMemory().record_plan_revision(
            revision_request.get("reason", "revise"),
            meta={"experiment": revision_request.get("experiment")},
        )
        return plan

    def _flag_unfalsifiable(self, plan: Plan) -> List[str]:
        flags = []
        contribs = plan.get("contributions") or []
        if not contribs and plan.get("expected_contributions"):
            for c in plan["expected_contributions"]:
                flags.append(f"Contribution lacks falsifiable prediction: {c}")
            return flags
        for c in contribs:
            if isinstance(c, str):
                flags.append(f"Contribution lacks falsifiable prediction: {c}")
                continue
            if not c.get("falsifiable_prediction"):
                flags.append(f"Missing falsifiable_prediction: {c.get('claim', c)}")
            if not c.get("statistical_test"):
                flags.append(f"Missing statistical_test: {c.get('claim', c)}")
        return flags

    def _flag_missing_baselines(self, plan: Plan) -> List[str]:
        flags = []
        for exp in plan.get("experiments") or []:
            baselines = exp.get("baselines") or []
            legacy = exp.get("baseline_comparison") or ""
            if not baselines and not str(legacy).strip():
                flags.append(f"Experiment '{exp.get('name')}' has no baselines")
            elif not baselines and legacy:
                # normalize later
                pass
        return flags

    def _ensure_falsifiable_contributions(self, plan: Plan, topic: Topic) -> List[Contribution]:
        existing = plan.get("contributions")
        if existing and isinstance(existing, list) and existing and isinstance(existing[0], dict):
            return existing
        claims = plan.get("expected_contributions") or [f"Advance {topic.get('title')}"]
        prompt = f"""
For each contribution claim, produce a falsifiable prediction and statistical test.

Claims: {json.dumps(claims)}
Topic: {topic.get('title')}

Return JSON array:
[{{"claim": "...", "falsifiable_prediction": "...", "statistical_test": "e.g. Welch t-test p<0.05 on accuracy vs baseline", "components": ["component_a"]}}]
"""
        parsed = parse_json_from_llm(call_llm(prompt, temperature=0.3, tier="strong"))
        if isinstance(parsed, list) and parsed:
            return parsed
        return [
            {
                "claim": c,
                "falsifiable_prediction": f"Method outperforms named baseline on primary metric",
                "statistical_test": "Welch t-test on metric across >=3 seeds, p<0.05",
                "components": ["main_method"],
            }
            for c in claims
        ]

    def _generate_plan_structure(self, topic: Topic, lessons: str) -> Plan:
        prompt = f"""
Create a detailed research plan.

Topic: {topic.get('title')}
Description: {topic.get('description')}
Rationale: {topic.get('rationale', 'N/A')}
Feasibility: {topic.get('feasibility', 5)}/10

Prior-run lessons:
{lessons}

Return JSON:
{{
  "title": "...",
  "sections": [{{"name": "Abstract", "content_requirements": "...", "key_points": [], "expected_length": "...", "dependencies": []}}],
  "research_questions": [],
  "methodology": "...",
  "expected_contributions": [],
  "compute_budget": "CPU-only sklearn-scale synthetic or small public datasets",
    "dataset_availability": "synthetic or clearly named public dataset",
    "dataset_catalog_choice": "bundled_synthetic|sklearn_iris|sklearn_digits"
}}
Include standard sections: Abstract, Introduction, Related Work, Methods, Experiments, Results, Discussion, Limitations, Conclusion.
"""
        try:
            parsed = parse_json_from_llm(call_llm(prompt, temperature=0.5, tier="strong"))
            if isinstance(parsed, dict) and "sections" in parsed:
                parsed["created_at"] = datetime.now().isoformat()
                parsed["topic"] = topic.get("title")
                parsed["domain"] = self.runtime_config.research_domain
                return parsed
        except Exception as e:
            log_agent_action("PlannerAgent", "plan_generation_error", {"error": str(e)})
        return self._create_fallback_plan(topic)

    def _generate_experiments(self, topic: Topic, plan: Plan) -> List[ExperimentSpec]:
        contribs = plan.get("contributions") or []
        prompt = f"""
Design experiments for:
Topic: {topic.get('title')}
Methodology: {plan.get('methodology')}
Contributions: {json.dumps(contribs)[:3000]}
Compute budget: {plan.get('compute_budget')}

Each experiment MUST include:
- name: short non-empty experiment name (REQUIRED key is "name", never "experiment_name")
- baselines: list of real comparison methods (not empty)
- falsifiable_prediction
- statistical_test
- variants: list of 2-3 alternative design objects, e.g. [{{"name": "Variant Name", "methodology": "..."}}] (MUST be objects with "name", NOT plain strings)
- claimed_components: for ablations
- evaluation_metrics

Return JSON array of experiments. Example object keys:
{{"name": "Baseline Comparison", "baselines": ["logistic_regression"], "evaluation_metrics": ["accuracy"],
  "falsifiable_prediction": "...", "statistical_test": "...", "variants": [{{"name": "Simpler Model", "methodology": "Linear baseline"}}], "claimed_components": []}}
"""
        try:
            parsed = parse_json_from_llm(call_llm(prompt, temperature=0.5, tier="strong"))
            if isinstance(parsed, list) and parsed:
                for exp in parsed:
                    if not exp.get("baselines") and exp.get("baseline_comparison"):
                        exp["baselines"] = [exp["baseline_comparison"]]
                return self._normalize_experiments(parsed)
        except Exception as e:
            log_agent_action("PlannerAgent", "experiment_generation_error", {"error": str(e)})
        return self._create_fallback_experiments(topic)

    @staticmethod
    def _normalize_experiment_name(experiment: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        """Align LLM aliases onto the ExperimentSpec contract field `name`."""
        exp = dict(experiment)
        name = exp.get("name")
        if not (isinstance(name, str) and name.strip()):
            for alias in ("experiment_name", "title", "experiment", "id", "variant_name", "variant"):
                alt = exp.get(alias)
                if isinstance(alt, str) and alt.strip():
                    exp["name"] = alt.strip()
                    break
        if isinstance(exp.get("name"), str):
            exp["name"] = exp["name"].strip()
        if not exp.get("name"):
            exp["name"] = f"experiment_{index + 1}"
        # Drop alias so downstream validators and contracts see one canonical key.
        exp.pop("experiment_name", None)
        exp.pop("variant_name", None)
        return exp

    @staticmethod
    def _normalize_experiments(experiments: List[ExperimentSpec]) -> List[ExperimentSpec]:
        normalized: List[ExperimentSpec] = []
        for index, experiment in enumerate(experiments or []):
            if not isinstance(experiment, dict):
                continue
            exp = PlannerAgent._normalize_experiment_name(experiment, index)
            for key in ("variants", "alternatives"):
                variants = exp.get(key)
                if not isinstance(variants, list):
                    continue
                fixed = []
                for variant_index, variant in enumerate(variants):
                    if isinstance(variant, dict):
                        norm_var = PlannerAgent._normalize_experiment_name(variant, variant_index)
                        if not norm_var.get("name"):
                            norm_var["name"] = f"{exp.get('name')}_variant_{variant_index + 1}"
                        if not norm_var.get("methodology") and not norm_var.get("description"):
                            norm_var["methodology"] = f"{exp.get('methodology', '')} (variant {variant_index + 1})".strip()
                        fixed.append(norm_var)
                    elif isinstance(variant, str) and variant.strip():
                        var_text = variant.strip()
                        fixed.append({
                            "name": f"{exp.get('name')}_variant_{variant_index + 1}",
                            "description": var_text,
                            "methodology": var_text,
                        })
                    elif variant is not None:
                        var_text = str(variant).strip()
                        fixed.append({
                            "name": f"{exp.get('name')}_variant_{variant_index + 1}",
                            "description": var_text,
                            "methodology": var_text,
                        })
                exp[key] = fixed
            normalized.append(exp)
        return normalized

    def _attach_variants(self, experiments: List[ExperimentSpec]) -> List[ExperimentSpec]:
        experiments = self._normalize_experiments(experiments)
        for exp in experiments:
            if not exp.get("variants"):
                exp["variants"] = [
                    {"name": f"{exp.get('name')}_variant_a", "methodology": exp.get("methodology", "") + " (simpler model)", "description": "simpler model"},
                    {"name": f"{exp.get('name')}_variant_b", "methodology": exp.get("methodology", "") + " (different features)", "description": "different features"},
                ]
            else:
                # Ensure existing variants in the list are all valid dict objects
                exp["variants"] = [
                    v if isinstance(v, dict) else {"name": f"{exp.get('name')}_variant_{i+1}", "methodology": str(v), "description": str(v)}
                    for i, v in enumerate(exp["variants"])
                ]
            if not exp.get("baselines"):
                exp["baselines"] = ["logistic_regression", "random_guess"]
            if not exp.get("evaluation_metrics"):
                exp["evaluation_metrics"] = ["accuracy"]
            if not exp.get("falsifiable_prediction"):
                exp["falsifiable_prediction"] = "Proposed method mean metric > best baseline mean across seeds"
            if not exp.get("statistical_test"):
                exp["statistical_test"] = "Welch t-test, p<0.05, n_seeds>=3"
        return self._normalize_experiments(experiments)

    def _repair_plan(self, plan, topic, unfalsifiable=None, missing_baselines=None, schema_errors=None):
        unfalsifiable = unfalsifiable or []
        missing_baselines = missing_baselines or []
        schema_errors = schema_errors or []
        prompt = f"""
Repair this plan to satisfy all validation requirements.
Schema errors: {schema_errors}
Unfalsifiable flags: {unfalsifiable}
Missing baselines: {missing_baselines}

Requirements:
- Every experiment MUST be an object with non-empty "name", "baselines", "evaluation_metrics", "falsifiable_prediction", "statistical_test"
- Every item in "variants" MUST be a JSON object with at least "name" and "methodology" (e.g. [{{"name": "Variant Name", "methodology": "..."}}]), NOT strings or flat lists.

Plan fragment: {json.dumps({'contributions': plan.get('contributions'), 'experiments': plan.get('experiments')}, default=str)[:5000]}
Return JSON {{"contributions": [...], "experiments": [...]}}
"""
        parsed = parse_json_from_llm(call_llm(prompt, temperature=0.3, tier="strong"))
        if isinstance(parsed, dict):
            plan.update({k: parsed[k] for k in ("contributions", "experiments") if k in parsed})
            plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])
            plan["experiments"] = self._attach_variants(plan.get("experiments") or [])
            plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])
        return plan

    def _generate_dependencies(self, plan: Plan) -> List[Dependency]:
        deps = []
        for section in plan.get("sections") or []:
            if section.get("name") in ("Methods", "Experiments", "Results"):
                deps.append({
                    "from": "Introduction",
                    "to": section["name"],
                    "type": "content_dependency",
                    "description": f'{section["name"]} builds on Introduction',
                })
        return deps

    def _generate_timeline(self, plan: Plan) -> Timeline:
        return {
            "phases": [
                {"name": "Planning", "duration": "1 week", "tasks": ["Literature", "Falsifiable claims"]},
                {"name": "Branch search", "duration": "1 week", "tasks": ["Cheap probes", "Promote winner"]},
                {"name": "Full experiments", "duration": "1-2 weeks", "tasks": ["Multi-seed", "Ablations"]},
                {"name": "Writing + verification", "duration": "1-2 weeks", "tasks": ["Draft", "Hard checks"]},
            ],
            "total_duration": "4-6 weeks",
        }

    def _store_plan(self, plan: Plan, topic: Topic):
        try:
            plan_path = f"{self.runtime_config.output_dir}/plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2, default=str)
            self.vector_memory.add_embedding(
                generate_embedding(json.dumps({"title": plan.get("title"), "topic": topic.get("title")})),
                {
                    "type": "research_plan",
                    "namespace": "planner",
                    "content_class": "generated_narrative",
                    "retrieval_eligible": False,
                    "agent": "PlannerAgent",
                    "outcome_status": "unknown",
                    "topic": topic.get("title"),
                    "plan_file": plan_path,
                },
            )
        except Exception as e:
            log_agent_action("PlannerAgent", "plan_storage_error", {"error": str(e)})

    def _create_fallback_plan(self, topic: Topic) -> Plan:
        return {
            "title": f"Research on {topic.get('title')}",
            "sections": [
                {"name": n, "content_requirements": n, "key_points": [], "expected_length": "1-3 pages", "dependencies": []}
                for n in [
                    "Abstract", "Introduction", "Related Work", "Methods",
                    "Experiments", "Results", "Discussion", "Limitations", "Conclusion",
                ]
            ],
            "research_questions": ["Does the proposed method outperform baselines?"],
            "methodology": "Controlled comparison on synthetic/public data with multi-seed stats",
            "expected_contributions": ["A method that beats named baselines on primary metric"],
            "compute_budget": "CPU sklearn-scale",
            "dataset_availability": "synthetic",
            "created_at": datetime.now().isoformat(),
            "topic": topic.get("title"),
            "domain": self.runtime_config.research_domain,
        }

    def _create_fallback_experiments(self, topic: Topic) -> List[ExperimentSpec]:
        return [
            {
                "name": "Baseline Comparison",
                "purpose": "Compare proposed method vs baselines",
                "methodology": "Train proposed model and baselines on same splits",
                "baselines": ["logistic_regression", "random_forest"],
                "baseline_comparison": "logistic_regression, random_forest",
                "evaluation_metrics": ["accuracy", "f1"],
                "falsifiable_prediction": "Proposed mean accuracy > best baseline mean accuracy",
                "statistical_test": "Welch t-test p<0.05 across 3 seeds",
                "claimed_components": ["proposed_model"],
                "code_requirements": "Python, sklearn, numpy",
                "data_requirements": "synthetic classification data",
                "variants": [],
            },
            {
                "name": "Ablation Study",
                "purpose": "Measure contribution of claimed components",
                "methodology": "Remove each component and measure degradation",
                "baselines": ["full_model"],
                "evaluation_metrics": ["accuracy_delta"],
                "falsifiable_prediction": "Removing claimed component reduces accuracy by >1pp",
                "statistical_test": "paired comparison across seeds",
                "claimed_components": ["proposed_model"],
                "code_requirements": "Python",
                "data_requirements": "same as main",
                "variants": [],
            },
        ]
