"""
PlannerAgent — falsifiable, revisable experiment designer.
Bidirectional Engineer → Planner revision path; baselines required.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import call_llm, generate_embedding, log_agent_action, parse_json_from_llm
from core.llm import get_llm_client
from core.memory import memory
from core.run_log import get_tracker, CrossRunMemory


class PlannerAgent:
    """Creates falsifiable research plans with baselines and experiment branches."""

    def __init__(self):
        self.client = get_llm_client()

    def create_plan(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        log_agent_action("PlannerAgent", "start_planning", {"topic": topic.get("title")})
        lessons = CrossRunMemory().lessons_for_prompt()

        plan = self._generate_plan_structure(topic, lessons)
        plan["contributions"] = self._ensure_falsifiable_contributions(plan, topic)
        plan["experiments"] = self._generate_experiments(topic, plan)
        plan["experiments"] = self._attach_variants(plan["experiments"])

        unfalsifiable = self._flag_unfalsifiable(plan)
        missing_baselines = self._flag_missing_baselines(plan)
        if unfalsifiable or missing_baselines:
            plan = self._repair_plan(plan, topic, unfalsifiable, missing_baselines)

        plan["dependencies"] = self._generate_dependencies(plan)
        plan["timeline"] = self._generate_timeline(plan)
        plan["unfalsifiable_flags"] = self._flag_unfalsifiable(plan)
        plan["missing_baseline_flags"] = self._flag_missing_baselines(plan)
        plan["revision_history"] = []

        self._store_plan(plan, topic)
        log_agent_action("PlannerAgent", "plan_created", {
            "sections": len(plan.get("sections", [])),
            "experiments": len(plan.get("experiments", [])),
            "unfalsifiable": len(plan["unfalsifiable_flags"]),
        })
        return plan

    def revise_plan(
        self,
        plan: Dict[str, Any],
        revision_request: Dict[str, Any],
        topic: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Bidirectional edge: Engineer requested a plan revision."""
        log_agent_action("PlannerAgent", "revise_plan", revision_request)
        tracker = get_tracker()
        if tracker:
            tracker.bump("plan_revisions")
            tracker.scratch("PlannerAgent", "revision", revision_request)

        prompt = f"""
Revise this research plan based on an Engineer failure.

Reason: {revision_request.get('reason')}
Detail: {revision_request.get('detail')}
Failed experiment: {revision_request.get('experiment')}

Current plan JSON:
{json.dumps({k: plan.get(k) for k in ('title', 'methodology', 'experiments', 'contributions', 'expected_contributions')}, indent=2)[:6000]}

Requirements:
- Fix unsupported assumptions / APIs / data requirements
- Every contribution needs falsifiable_prediction + statistical_test
- Every experiment needs at least one real baseline in "baselines" list
- Provide 2-3 "variants" (cheap alternative designs) per experiment
Return full updated plan fragment as JSON with keys: methodology, contributions, experiments, revision_notes
"""
        response = call_llm(prompt, temperature=0.4, tier="strong")
        parsed = parse_json_from_llm(response) or {}
        if isinstance(parsed, dict):
            for key in ("methodology", "contributions", "experiments"):
                if key in parsed:
                    plan[key] = parsed[key]
            plan.setdefault("revision_history", []).append({
                "request": revision_request,
                "notes": parsed.get("revision_notes", ""),
                "at": datetime.now().isoformat(),
            })
        CrossRunMemory().record_plan_revision(
            revision_request.get("reason", "revise"),
            meta={"experiment": revision_request.get("experiment")},
        )
        return plan

    def _flag_unfalsifiable(self, plan: Dict[str, Any]) -> List[str]:
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

    def _flag_missing_baselines(self, plan: Dict[str, Any]) -> List[str]:
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

    def _ensure_falsifiable_contributions(self, plan: Dict[str, Any], topic: Dict[str, Any]) -> List[Dict]:
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

    def _generate_plan_structure(self, topic: Dict[str, Any], lessons: str) -> Dict[str, Any]:
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
  "dataset_availability": "synthetic or clearly named public dataset"
}}
Include standard sections: Abstract, Introduction, Related Work, Methods, Experiments, Results, Discussion, Limitations, Conclusion.
"""
        try:
            parsed = parse_json_from_llm(call_llm(prompt, temperature=0.5, tier="strong"))
            if isinstance(parsed, dict) and "sections" in parsed:
                parsed["created_at"] = datetime.now().isoformat()
                parsed["topic"] = topic.get("title")
                parsed["domain"] = config.research_domain
                return parsed
        except Exception as e:
            log_agent_action("PlannerAgent", "plan_generation_error", {"error": str(e)})
        return self._create_fallback_plan(topic)

    def _generate_experiments(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        contribs = plan.get("contributions") or []
        prompt = f"""
Design experiments for:
Topic: {topic.get('title')}
Methodology: {plan.get('methodology')}
Contributions: {json.dumps(contribs)[:3000]}
Compute budget: {plan.get('compute_budget')}

Each experiment MUST include:
- baselines: list of real comparison methods (not empty)
- falsifiable_prediction
- statistical_test
- variants: 2-3 cheap alternative designs
- claimed_components: for ablations
- evaluation_metrics

Return JSON array of experiments.
"""
        try:
            parsed = parse_json_from_llm(call_llm(prompt, temperature=0.5, tier="strong"))
            if isinstance(parsed, list) and parsed:
                for exp in parsed:
                    if not exp.get("baselines") and exp.get("baseline_comparison"):
                        exp["baselines"] = [exp["baseline_comparison"]]
                return parsed
        except Exception as e:
            log_agent_action("PlannerAgent", "experiment_generation_error", {"error": str(e)})
        return self._create_fallback_experiments(topic)

    def _attach_variants(self, experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for exp in experiments:
            if not exp.get("variants"):
                exp["variants"] = [
                    {**exp, "name": f"{exp.get('name')}_variant_a", "methodology": exp.get("methodology", "") + " (simpler model)"},
                    {**exp, "name": f"{exp.get('name')}_variant_b", "methodology": exp.get("methodology", "") + " (different features)"},
                ]
            if not exp.get("baselines"):
                exp["baselines"] = ["logistic_regression", "random_guess"]
            if not exp.get("falsifiable_prediction"):
                exp["falsifiable_prediction"] = "Proposed method mean metric > best baseline mean across seeds"
            if not exp.get("statistical_test"):
                exp["statistical_test"] = "Welch t-test, p<0.05, n_seeds>=3"
        return experiments

    def _repair_plan(self, plan, topic, unfalsifiable, missing_baselines):
        prompt = f"""
Repair this plan.
Unfalsifiable flags: {unfalsifiable}
Missing baselines: {missing_baselines}
Plan fragment: {json.dumps({'contributions': plan.get('contributions'), 'experiments': plan.get('experiments')}, default=str)[:5000]}
Return JSON {{"contributions": [...], "experiments": [...]}}
"""
        parsed = parse_json_from_llm(call_llm(prompt, temperature=0.3, tier="strong"))
        if isinstance(parsed, dict):
            plan.update({k: parsed[k] for k in ("contributions", "experiments") if k in parsed})
            plan["experiments"] = self._attach_variants(plan.get("experiments") or [])
        return plan

    def _generate_dependencies(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
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

    def _generate_timeline(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "phases": [
                {"name": "Planning", "duration": "1 week", "tasks": ["Literature", "Falsifiable claims"]},
                {"name": "Branch search", "duration": "1 week", "tasks": ["Cheap probes", "Promote winner"]},
                {"name": "Full experiments", "duration": "1-2 weeks", "tasks": ["Multi-seed", "Ablations"]},
                {"name": "Writing + verification", "duration": "1-2 weeks", "tasks": ["Draft", "Hard checks"]},
            ],
            "total_duration": "4-6 weeks",
        }

    def _store_plan(self, plan: Dict[str, Any], topic: Dict[str, Any]):
        try:
            plan_path = f"{config.output_dir}/plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2, default=str)
            memory.add_embedding(
                generate_embedding(json.dumps({"title": plan.get("title"), "topic": topic.get("title")})),
                {"type": "research_plan", "topic": topic.get("title"), "plan_file": plan_path},
            )
        except Exception as e:
            log_agent_action("PlannerAgent", "plan_storage_error", {"error": str(e)})

    def _create_fallback_plan(self, topic: Dict[str, Any]) -> Dict[str, Any]:
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
            "domain": config.research_domain,
        }

    def _create_fallback_experiments(self, topic: Dict[str, Any]) -> List[Dict[str, Any]]:
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
