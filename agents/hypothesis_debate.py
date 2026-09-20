"""
Hypothesis Debate — multi-round adversarial debate with ensemble judging + Elo.
"""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import log_agent_action, parse_json_from_llm, is_degenerate_llm_output, call_llm_json
from core.llm import call_llm
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.memory import memory
from core.run_log import get_tracker
from core.capabilities import SANDBOX_CAPABILITY_MANIFEST, check_plan_feasibility
from core.evidence_synthesis import validate_topic_admission

logger = logging.getLogger(__name__)

REVIEWER_CHECKLIST = [
    "soundness",
    "significance",
    "reproducibility",
    "ethics",
    "novelty",
    "feasibility",
]


def _coerce_int(value: Any, default: int = 0) -> int:
    """int() that treats missing/None as default (dict.get default does not)."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class DebateResult:
    topic: str
    proposer_argument: str
    challenger_argument: str
    moderator_decision: str
    score: float
    passed: bool
    reasoning: str
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    objections: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_objections: List[str] = field(default_factory=list)
    ensemble_scores: List[float] = field(default_factory=list)
    elo_delta: float = 0.0


class EloStore:
    """Backward-compatible Elo store with observation counts and shrinkage.

    Legacy files that store `{kind: float}` or partial dicts (missing/null
    observations) are upgraded on read and rewritten once so new fields are
    never read as None. Selection uses the shrunk rating; raw_rating is
    preserved for audit.
    """

    def __init__(self, path: Optional[str] = None, context: Optional[RunContext] = None):
        runtime_config = (context or get_active_context()).config if (context or get_active_context()) else config
        self.path = Path(path or runtime_config.elo_ratings_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prior = float(getattr(runtime_config, "elo_prior_rating", 1500.0))
        self.shrinkage_k = float(getattr(runtime_config, "elo_shrinkage_k", 8.0))
        self.min_observations = _coerce_int(getattr(runtime_config, "elo_min_observations", 5), 5)
        self.records: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.records = self._upgrade(raw)
                if self._needs_migration(raw):
                    self._persist()
            except Exception as e:
                logger.warning("EloStore corrupted, resetting: %s", e)
                self.records = {}

    @staticmethod
    def _needs_migration(raw: Dict[str, Any]) -> bool:
        """True when on-disk shape is legacy flat or missing/null observation fields."""
        for value in (raw or {}).values():
            if not isinstance(value, dict):
                return True
            if "rating" not in value:
                return True
            if "observations" not in value and "observation_count" not in value:
                return True
            if value.get("observations") is None and value.get("observation_count") is None:
                return True
            if "raw_rating" not in value or "shrinkage_k" not in value:
                return True
        return False

    def _persist(self) -> None:
        self.path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    def _upgrade(self, raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        upgraded: Dict[str, Dict[str, Any]] = {}
        for key, value in (raw or {}).items():
            if isinstance(value, dict) and ("rating" in value or "raw_rating" in value):
                rating = _coerce_float(value.get("rating", value.get("raw_rating")), self.prior)
                # Accept both observations and legacy observation_count aliases.
                observations = _coerce_int(
                    value.get("observations", value.get("observation_count")),
                    0,
                )
                record = {
                    "rating": rating,
                    "raw_rating": _coerce_float(value.get("raw_rating", rating), rating),
                    "observations": observations,
                    "prior": _coerce_float(value.get("prior"), self.prior),
                    "shrinkage_k": _coerce_float(value.get("shrinkage_k"), self.shrinkage_k),
                    "updated_at": value.get("updated_at"),
                }
            else:
                # Legacy flat float
                rating = _coerce_float(value, self.prior)
                record = {
                    "rating": rating,
                    "raw_rating": rating,
                    "observations": 0,
                    "prior": self.prior,
                    "shrinkage_k": self.shrinkage_k,
                    "updated_at": None,
                }
            upgraded[key] = record
        return upgraded

    @property
    def ratings(self) -> Dict[str, float]:
        """Shrunk ratings used for selection (legacy-compatible mapping)."""
        return {key: float(rec["rating"]) for key, rec in self.records.items()}

    def _shrink(self, raw_rating: float, observations: int, prior: float, k: float) -> float:
        # Pull toward prior until enough observations accumulate.
        weight = observations / (observations + k) if (observations + k) else 0.0
        return prior + (raw_rating - prior) * weight

    def get_record(self, key: str) -> Dict[str, Any]:
        if key in self.records:
            return dict(self.records[key])
        kind = hypothesis_kind(key)
        if kind in self.records:
            return dict(self.records[kind])
        return {
            "rating": self.prior,
            "raw_rating": self.prior,
            "observations": 0,
            "prior": self.prior,
            "shrinkage_k": self.shrinkage_k,
            "updated_at": None,
        }

    def get(self, key: str, default: Optional[float] = None) -> float:
        if key in self.records:
            return float(self.records[key]["rating"])
        kind = hypothesis_kind(key)
        if kind in self.records:
            return float(self.records[kind]["rating"])
        return float(self.prior if default is None else default)

    def observations(self, key: str) -> int:
        return _coerce_int(self.get_record(key).get("observations"), 0)

    def under_observed_kinds(self, kinds: List[str]) -> List[str]:
        return [k for k in kinds if self.observations(k) < self.min_observations]

    def update(self, hypothesis_key: str, score: float, passed: bool) -> float:
        """Elo vs a fixed reviewer bar; persist raw and shrunk ratings separately."""
        kind = hypothesis_kind(hypothesis_key)
        record = self.get_record(kind)
        ra = _coerce_float(record.get("raw_rating"), self.prior)
        rb = self.prior
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        outcome = 1.0 if passed else (0.5 if score >= 6 else 0.0)
        k_factor = 32
        delta = k_factor * (outcome - ea)
        new_raw = ra + delta
        observations = _coerce_int(record.get("observations"), 0) + 1
        prior = _coerce_float(record.get("prior"), self.prior)
        shrink_k = _coerce_float(record.get("shrinkage_k"), self.shrinkage_k)
        # Until min observations, force stronger shrinkage toward prior.
        effective_k = (
            max(shrink_k, self.min_observations - observations + 1)
            if observations < self.min_observations
            else shrink_k
        )
        shrunk = self._shrink(new_raw, observations, prior, effective_k)
        self.records[kind] = {
            "rating": shrunk,
            "raw_rating": new_raw,
            "observations": observations,
            "prior": prior,
            "shrinkage_k": shrink_k,
            "updated_at": datetime.now().isoformat(),
        }
        self._persist()
        logger.info("Elo updated: kind=%s score=%.1f passed=%s raw_rating=%.1f->%.1f shrunk=%.1f delta=%.1f observations=%d",
                     kind, score, passed, ra, new_raw, shrunk, delta, observations)
        return delta

    def record_strategy_outcome(self, strategy: str, outcome_status: str) -> float:
        """Feature 8: record seed-strategy outcome under a strategy:<name> namespace key,
        reusing the existing rating/shrinkage record shape. Returns Elo delta.

        outcome_status uses debate-derived labels: "debate_pass", "debate_weak",
        "debate_fail". These are NOT experiment-level outcomes — kind-Elo uses
        the same semantics (there is no post-experiment Elo update in the system).
        """
        key = f"strategy:{strategy}"
        record = self.get_record(key)
        ra = _coerce_float(record.get("raw_rating"), self.prior)
        rb = self.prior
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        outcome_map = {"debate_pass": 1.0, "debate_weak": 0.5, "debate_fail": 0.0}
        outcome = outcome_map.get(outcome_status, 0.5)
        k_factor = 32
        delta = k_factor * (outcome - ea)
        new_raw = ra + delta
        observations = _coerce_int(record.get("observations"), 0) + 1
        prior = _coerce_float(record.get("prior"), self.prior)
        shrink_k = _coerce_float(record.get("shrinkage_k"), self.shrinkage_k)
        effective_k = (
            max(shrink_k, self.min_observations - observations + 1)
            if observations < self.min_observations
            else shrink_k
        )
        shrunk = self._shrink(new_raw, observations, prior, effective_k)
        self.records[key] = {
            "rating": shrunk,
            "raw_rating": new_raw,
            "observations": observations,
            "prior": prior,
            "shrinkage_k": shrink_k,
            "updated_at": datetime.now().isoformat(),
        }
        self._persist()
        logger.info("Elo strategy outcome: strategy=%s outcome=%s delta=%.1f new_rating=%.1f", strategy, outcome_status, delta, shrunk)
        return delta


def hypothesis_kind(title: str) -> str:
    """Coarse bucket for Elo (kinds of hypotheses)."""
    t = title.lower()
    for kind in ("attention", "graph", "diffusion", "reinforcement", "federated", "llm", "vision", "nlp"):
        if kind in t:
            return kind
    return "general"


def _as_dict(value: Any) -> Dict[str, Any]:
    """Return value when it is a dict, else {} — guards .get against lists."""
    return value if isinstance(value, dict) else {}


def _normalize_objection_payload(parsed: Any) -> Dict[str, Any]:
    """Coerce LLM objection payloads into the canonical envelope shape.

    Models frequently return a bare array of objections or a single objection
    object instead of the requested {"objections": [...]}. Both are valid
    content in the wrong shape — coerce instead of crashing on ``.get``.
    """
    if isinstance(parsed, list):
        parsed = {"objections": parsed}
    if not isinstance(parsed, dict):
        return {}
    raw_objections = parsed.get("objections")
    if isinstance(raw_objections, dict):
        parsed["objections"] = [raw_objections]
    return parsed


def _ensure_objections_are_dicts(objections: List[Any]) -> List[Dict[str, Any]]:
    """Defensive normalizer: ensure every item in an objections list is a dict.
    LLMs occasionally return nested lists or bare strings inside objection lists.
    This prevents 'list object has no attribute get' crashes downstream."""
    out: List[Dict[str, Any]] = []
    for item in objections:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, list):
            # Nested list — flatten: take first dict found, or wrap
            for sub in item:
                if isinstance(sub, dict):
                    out.append(sub)
                    break
            else:
                out.append({"objection": str(item), "severity": 3, "status": "unresolved"})
        elif isinstance(item, str):
            out.append({"objection": item, "severity": 3, "status": "unresolved"})
    return out


class ProposerAgent:
    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()
        self.vector_memory = self.context.memory if self.context else memory

    def _prior_objection_tags(self) -> List[Dict[str, Any]]:
        """Structured objection tags from prior debates so round-1 arguments
        can preempt historically recurring failure modes."""
        rows = self.vector_memory.get_prompt_context(namespace="debate_transcripts", k=8)
        tags = []
        for row in rows:
            signal = row.get("signal") or {}
            tags.append({
                "objection_type": signal.get("objection_type") or signal.get("objection_types"),
                "severity": signal.get("severity"),
                "resolution_status": signal.get("resolution_status"),
            })
        return tags

    def _evidence_context(self, topic: Dict[str, Any], max_papers: int = 5, snippet_chars: int = 350) -> str:
        """Render the topic's retrieved literature evidence for grounded arguing."""
        papers = topic.get("literature_evidence") or []
        slim = []
        for paper in papers[:max_papers]:
            if not isinstance(paper, dict):
                continue
            slim.append({
                "title": paper.get("title", ""),
                "abstract": (paper.get("abstract") or "")[:snippet_chars],
            })
        slim = [p for p in slim if p["title"] or p["abstract"]]
        return json.dumps(slim, default=str)[:4000] if slim else "none retrieved"

    def build_argument(self, topic: Dict[str, Any]) -> str:
        logger.info("Proposer generating argument for: %s", topic.get("title", "<untitled>"))
        prior_tags = self._prior_objection_tags()
        structured_hyp = _as_dict(topic.get("structured_hypothesis"))
        prompt = f"""
You are a research proposer. Build a compelling, realistic argument.
Topic: {topic.get('title')}
Description: {topic.get('description')}
Rationale: {topic.get('rationale', 'N/A')}
Impact: {topic.get('impact', 'N/A')}
Feasibility: {topic.get('feasibility', 5)}/10

Structured Hypothesis Contract (authoritative):
{json.dumps(topic.get('structured_hypothesis') or {}, indent=2, default=str)[:3000]}

Retrieved literature evidence (ground every novelty/gap claim in this):
{self._evidence_context(topic)}

Historically recurring objection tags from prior debates (preempt these proactively):
{json.dumps(self._prior_objection_tags(), default=str)[:1200]}

Rules:
- Ground every novelty claim in the retrieved evidence above; explicitly name the
  closest prior work and state what remains untested.
- Address the recurring objection types listed above before they are raised.
- Do not claim evidence you do not have; describe what will be measured and how.
Include: hypothesis, theory, evidence, novelty, methodology, expected outcomes, falsifiable prediction.
"""
        arg = ""
        for attempt in range(2):
            raw = call_llm(prompt, temperature=0.7, tier="strong")
            if not is_degenerate_llm_output(raw, min_chars=30):
                arg = raw
                break
            log_agent_action("ProposerAgent", "degenerate_argument_retry", {"attempt": attempt + 1})

        if not arg:
            arg = "PROPOSER_ARGUMENT_INVALID"
            log_agent_action("ProposerAgent", "argument_failed_closed", {"topic": topic.get("title")})
        else:
            log_agent_action("ProposerAgent", "built_argument", {"len": len(arg)})
        return arg

    def respond_to_objections(
        self,
        topic: Dict[str, Any],
        prior_argument: str,
        objections: List[Dict[str, Any]],
    ) -> str:
        """Respond to EACH specific objection (not a global rebuttal)."""
        logger.info("Proposer responding to %d objections for: %s", len(objections), topic.get("title", "<untitled>"))
        obj_text = "\n".join(
            f"- [{o.get('criterion')}] {o.get('objection')}" for o in objections
        )
        prompt = f"""
Respond point-by-point to each Challenger objection. Do not give a vague global rebuttal.
Topic: {topic.get('title')}
Your prior argument: {prior_argument[:3000]}
Objections:
{obj_text}

Format:
For each objection: OBJECTION: ... RESPONSE: ... CONCESSION (if any): ...
"""
        response = ""
        for attempt in range(2):
            raw = call_llm(prompt, temperature=0.5, tier="strong")
            if not is_degenerate_llm_output(raw, min_chars=20):
                response = raw
                break
            log_agent_action("ProposerAgent", "degenerate_response_retry", {"attempt": attempt + 1})

        return response or "PROPOSER_RESPONSE_INVALID"


class ChallengerAgent:
    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()
        self.vector_memory = self.context.memory if self.context else memory

    def _prior_objection_tags(self) -> List[Dict[str, Any]]:
        """Retrieve only structured objection tags — never prior argument prose."""
        mem = getattr(self, "vector_memory", None) or (
            self.context.memory if getattr(self, "context", None) else memory
        )
        rows = mem.get_prompt_context(namespace="debate_transcripts", k=8)
        tags = []
        for row in rows:
            signal = row.get("signal") or {}
            tags.append({
                "objection_type": signal.get("objection_type") or signal.get("objection_types"),
                "severity": signal.get("severity"),
                "resolution_status": signal.get("resolution_status"),
                "run_id": row.get("run_id"),
                "outcome_status": row.get("outcome_status"),
            })
        return tags

    def build_rebuttal(self, topic: Dict[str, Any], proposer_argument: str) -> str:
        """Checklist-grounded critique — attacks scientific validity, confounders, baselines, and feasibility."""
        logger.info("Challenger generating rebuttal for: %s", topic.get("title", "<untitled>"))
        rating = float(topic.get("elo_rating", EloStore(context=self.context).get(hypothesis_kind(topic.get("title", "")))))
        kind = topic.get("hypothesis_kind") or hypothesis_kind(topic.get("title", ""))
        history_note = (
            f"This is a {kind} hypothesis with a low historical Elo ({rating:.0f}). "
            "Scrutinize the failure modes that have historically made this kind fail."
            if rating < 1450 else
            f"This is a {kind} hypothesis with historical Elo {rating:.0f}; still independently verify its claims."
        )
        prior_tags = self._prior_objection_tags()
        structured_hyp = _as_dict(topic.get("structured_hypothesis"))
        prompt = f"""
You are an adversarial scientific reviewer. Your goal is to find the strongest legitimate reasons this hypothesis may be flawed, unfalsifiable, confounded, or unexecutable.

Topic: {topic.get('title')}
Structured Hypothesis Contract:
{json.dumps(structured_hyp, indent=2, default=str)[:3000]}

Proposer Argument:
{proposer_argument[:3000]}

Historical signal: {history_note}
Prior objection tags only (never reuse prior prose): {json.dumps(prior_tags)[:1500]}
Sandbox capability manifest: {json.dumps(SANDBOX_CAPABILITY_MANIFEST.as_dict(), sort_keys=True)}

Scrutinize:
1. Hypothesis clarity & falsifiability (is the falsification condition concrete and measurable?)
2. Confounders & competing explanations (could an alternative mechanism produce the same result?)
3. Baseline adequacy (are meaningful comparison baselines included?)
4. Evaluation & metric adequacy (can the chosen metrics detect the proposed effect?)
5. Statistical adequacy & power (sample size, seed count, statistical test)
6. Feasibility vs Sandbox Manifest and Local Dataset Catalog (local datasets only, no downloads, CPU limits)
7. Minimum Viable Experiment sufficiency (does the experiment distinguish the hypothesis from competing explanations?)

Return JSON:
{{
  "objections": [
    {{
      "criterion": "confounder|baseline|falsifiability|soundness|feasibility|evaluation|statistical|novelty",
      "objection": "...",
      "severity": 1-5,
      "status": "unresolved",
      "required_resolution": "...",
      "source": "challenger_audit"
    }}
  ],
  "summary_rebuttal": "..."
}}
"""
        self._challenger_invalid = False
        objections: List[Dict[str, Any]] = []
        summary = ""
        raw = ""
        attempt_state = {"n": 0}

        def _rebuttal_call(current_prompt: str, **kwargs: Any) -> str:
            nonlocal raw
            raw = call_llm(current_prompt, **kwargs)
            attempt_state["n"] += 1
            if is_degenerate_llm_output(raw, min_chars=20):
                log_agent_action("ChallengerAgent", "degenerate_rebuttal", {
                    "attempt": attempt_state["n"],
                    "preview": str(raw)[:120],
                })
                return ""
            return raw

        # Self-correcting parse: on malformed output the model is re-asked with
        # the parse error and its own offending excerpt (same protocol as the
        # screener/followup paths) instead of a parse-blind identical re-roll.
        parsed = call_llm_json(
            prompt,
            temperature=0.5,
            tier="strong",
            attempts=2,
            max_tokens=16384,
            call_fn=_rebuttal_call,
        )
        # Fallback: if structured parse failed, try raw JSON extraction on the
        # last LLM response (reasoning models often emit valid JSON after
        # chain-of-thought that call_llm_json's stricter path misses).
        if parsed is None and raw:
            parsed = parse_json_from_llm(raw)
        parsed = _normalize_objection_payload(parsed)
        raw_objections = parsed.get("objections")
        if isinstance(raw_objections, list) and len(raw_objections) > 0:
            for obj in raw_objections:
                if isinstance(obj, dict):
                    obj.setdefault("status", "unresolved")
                    obj.setdefault("source", "challenger_audit")
                    objections.append(obj)
            summary = str(parsed.get("summary_rebuttal") or raw)
        else:
            # Fail closed: never treat empty/garbled challenger output as zero objections.
            self._challenger_invalid = True
            objections = [{
                "criterion": "soundness",
                "objection": "Challenger produced empty or garbled output; debate fails closed.",
                "severity": 5,
                "status": "unresolved",
                "required_resolution": "Challenger must produce valid critique.",
                "source": "challenger_output_guard",
            }]
            summary = "CHALLENGER_OUTPUT_INVALID"
            log_agent_action("ChallengerAgent", "rebuttal_invalid_after_retry", {
                "raw_preview": str(raw)[:200],
                "parsed_shape": type(parsed).__name__,
            })

        feasibility_errors = check_plan_feasibility(
            {"methodology": topic.get("description", ""), "experiments": [{"dataset": {"name": topic.get("dataset_plan", "")}}]},
            SANDBOX_CAPABILITY_MANIFEST,
        )
        objections.extend({
            "criterion": "feasibility",
            "objection": reason,
            "severity": 5,
            "status": "unresolved",
            "required_resolution": "Rescope experiment within sandbox capabilities",
            "source": "capability_manifest",
        } for reason in feasibility_errors)

        # Minimum viable experiment validation
        mve_check = self.validate_minimum_experiment(topic)
        if not mve_check.get("is_discriminative", True) or not mve_check.get("is_executable", True):
            for note in mve_check.get("validation_notes", []):
                objections.append({
                    "criterion": "soundness",
                    "objection": f"Minimum viable experiment flaw: {note}",
                    "severity": 4,
                    "status": "unresolved",
                    "required_resolution": "Refine MVE to be executable and discriminative",
                    "source": "minimum_experiment_check",
                })

        self._last_objections = objections
        log_agent_action("ChallengerAgent", "built_rebuttal", {
            "n_objections": len(objections),
            "invalid": bool(self._challenger_invalid),
        })
        return summary if isinstance(summary, str) else json.dumps(parsed)

    def validate_minimum_experiment(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that the minimum viable experiment is both executable and scientifically discriminative."""
        structured_hyp = _as_dict(topic.get("structured_hypothesis"))
        mve = structured_hyp.get("minimum_viable_experiment") or {}
        notes: List[str] = []
        is_executable = True
        is_discriminative = True

        if not mve:
            return {
                "is_executable": True,
                "is_discriminative": True,
                "validation_notes": [],
            }

        # Executability check against local catalog
        from core.datasets import list_datasets
        catalog_names = {d["name"] for d in list_datasets()}
        dataset_name = mve.get("dataset", "bundled_synthetic")
        if dataset_name not in catalog_names and "synthetic" not in dataset_name.lower():
            is_executable = False
            notes.append(f"Dataset '{dataset_name}' not found in local catalog and not synthetic")

        # Discriminative power check
        competing = structured_hyp.get("competing_explanations") or []
        falsification = structured_hyp.get("falsification_condition") or mve.get("falsification_test")
        if not falsification:
            is_discriminative = False
            notes.append("No defined falsification test in minimum viable experiment")
        if competing and not mve.get("baseline") and not mve.get("models"):
            is_discriminative = False
            notes.append("MVE lacks baselines to distinguish hypothesis from competing explanations")

        return {
            "is_executable": is_executable,
            "is_discriminative": is_discriminative,
            "validation_notes": notes,
        }

    def followup_objections(
        self,
        topic: Dict[str, Any],
        proposer_response: str,
        prior_objections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Track objections across rounds with invariant: no objection may silently disappear."""
        logger.info("Challenger evaluating followup objections for: %s (%d prior objections)",
                     topic.get("title", "<untitled>"), len(prior_objections))
        prompt = f"""
Given the proposer's point-by-point responses, evaluate each prior objection.
Prior objections:
{json.dumps(prior_objections, indent=2)[:3000]}

Proposer responses:
{proposer_response[:3000]}

For EACH prior objection, decide if it is resolved or remains unresolved.
Return JSON:
{{
  "objections": [
    {{
      "criterion": "...",
      "objection": "...",
      "severity": 3,
      "status": "resolved|unresolved",
      "resolution": "...",
      "required_resolution": "..."
    }}
  ]
}}
"""
        for attempt in range(2):
            parsed = call_llm_json(prompt, temperature=0.3, tier="judge", attempts=2, call_fn=call_llm)
            # A bare array of objections is valid content in the wrong envelope.
            if isinstance(parsed, list):
                parsed = {"objections": parsed}
            if not isinstance(parsed, dict):
                log_agent_action("ChallengerAgent", "degenerate_followup", {"attempt": attempt + 1})
                continue
            raw_objs = parsed.get("objections")
            if isinstance(raw_objs, list) and len(raw_objs) > 0:
                # Merge status onto prior objections to ensure no objection is dropped
                updated = []
                seen_texts = set()
                for obj in raw_objs:
                    if isinstance(obj, dict):
                        text = str(obj.get("objection", ""))
                        seen_texts.add(text.lower().strip())
                        updated.append(obj)
                    elif isinstance(obj, str):
                        updated.append({"objection": obj, "severity": 3, "status": "unresolved"})
                        seen_texts.add(obj.lower().strip())
                    elif isinstance(obj, list) and obj:
                        # Nested list — take first dict or wrap
                        for sub in obj:
                            if isinstance(sub, dict):
                                text = str(sub.get("objection", ""))
                                seen_texts.add(text.lower().strip())
                                updated.append(sub)
                                break
                        else:
                            updated.append({"objection": str(obj), "severity": 3, "status": "unresolved"})
                for prior in prior_objections:
                    if not isinstance(prior, dict):
                        prior = {"objection": str(prior), "severity": 3}
                    p_text = str(prior.get("objection", "")).lower().strip()
                    if p_text not in seen_texts:
                        # Prior objection was dropped by LLM; preserve as unresolved
                        unresolved_copy = dict(prior)
                        unresolved_copy["status"] = "unresolved"
                        updated.append(unresolved_copy)
                return updated

        # Fail closed: preserve all prior objections as unresolved
        return [
            dict(obj, status="unresolved") if isinstance(obj, dict)
            else {"objection": str(obj), "severity": 3, "status": "unresolved"}
            for obj in prior_objections
        ]


class ModeratorAgent:
    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.runtime_config = self.context.config if self.context else config
        self.client = get_llm_client()

    def evaluate_debate(
        self,
        topic: Dict[str, Any],
        rounds: List[Dict[str, Any]],
        objections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Ensemble judge: hard deterministic gates dominate; soft scores cannot rescue hard fails."""
        logger.info("Moderator evaluating debate for: %s (%d rounds, %d unresolved objections)",
                     topic.get("title", "<untitled>"), len(rounds),
                     len([o for o in objections if isinstance(o, dict) and o.get("status") != "resolved"]))
        models = self._judge_models()
        scores: List[float] = []
        reasonings: List[str] = []
        valid_judge_responses = 0
        transcript = json.dumps(rounds, default=str)[:6000]

        unresolved = [
            o for o in objections if isinstance(o, dict) and o.get("status") != "resolved"
        ]

        # Hard Gates evaluation
        hard_gates = {
            "hypothesis_falsifiable": bool(
                topic.get("falsifiable_prediction")
                or _as_dict(topic.get("structured_hypothesis")).get("falsification_condition")
            ),
            "measurable_outcome_exists": bool(
                _as_dict(topic.get("structured_hypothesis")).get("dependent_variables")
                or topic.get("rationale")
            ),
            "no_severe_unresolved_objections": not any(
                _coerce_int(u.get("severity"), 0) >= 4 for u in unresolved
            ),
            "resources_available": not any(
                u.get("criterion") == "feasibility" and _coerce_int(u.get("severity"), 0) >= 4 for u in unresolved
            ),
        }
        all_hard_passed = all(hard_gates.values())

        base_prompt = f"""
Moderate this multi-round research debate.
Topic: {topic.get('title')}
Structured Hypothesis: {json.dumps(topic.get('structured_hypothesis', {}), default=str)[:1500]}
Transcript: {transcript}
Unresolved objections: {json.dumps(unresolved, default=str)[:2000]}
Pass threshold: {self.runtime_config.debate_pass_threshold}

Score overall scientific merit 1-10 on soft dimensions (significance, novelty, clarity, expected contribution).
JSON: {{"score": 7.5, "passed": true, "reasoning": "...", "decision": "PASS|FAIL"}}
"""
        for model in models:
            raw = call_llm(base_prompt, temperature=0.2, tier="judge", model=model)
            if is_degenerate_llm_output(raw, min_chars=20):
                continue
            parsed = parse_json_from_llm(raw) or {}
            if isinstance(parsed, dict) and "score" in parsed:
                try:
                    score_val = float(parsed["score"])
                    scores.append(score_val)
                    reasonings.append(str(parsed.get("reasoning", raw[:300])))
                    valid_judge_responses += 1
                except (TypeError, ValueError):
                    pass

        if valid_judge_responses == 0:
            # Fail closed: missing judge output must never become a passing score.
            return {
                "score": 0.0,
                "passed": False,
                "ensemble_scores": [],
                "disagreement": 0.0,
                "needs_longer_debate": False,
                "reasoning": "No valid judge responses obtained; debate fails closed.",
                "decision": "FAIL",
                "valid_judge_responses": 0,
                "hard_gates": hard_gates,
            }

        mean = sum(scores) / len(scores)
        disagreement = max(scores) - min(scores) if len(scores) > 1 else 0.0
        agreed = disagreement <= 2.5  # Relaxed from 1.5 to allow more diversity in judge scores

        # Key rule: soft score cannot override hard FAIL
        passed = all_hard_passed and agreed and (mean >= self.runtime_config.debate_pass_threshold)

        logger.info("Moderator decision: score=%.2f passed=%s disagreement=%.2f hard_gates=%s judge_responses=%d",
                     mean, passed, disagreement, all_hard_passed, valid_judge_responses)

        return {
            "score": mean,
            "passed": passed,
            "ensemble_scores": scores,
            "disagreement": disagreement,
            "needs_longer_debate": disagreement > 2.5 and len(rounds) < self.runtime_config.debate_max_rounds,
            "reasoning": " | ".join(reasonings[:3]),
            "decision": "PASS" if passed else "FAIL",
            "valid_judge_responses": valid_judge_responses,
            "hard_gates": hard_gates,
            "unresolved_objections": unresolved,
        }

    def _judge_models(self) -> List[str]:
        models = self.runtime_config.get_ensemble_models()
        if models:
            return models[:3]
        out = []
        for tier in ("judge", "strong", "cheap"):
            m = self.runtime_config.resolve_model(tier)
            if m and m not in out:
                out.append(m)
        return out[:3] or [self.runtime_config.resolve_model("default")]


class HypothesisDebateSystem:
    def __init__(self, context: Optional[RunContext] = None):
        context = context or get_active_context()
        self.context = context
        self.runtime_config = context.config if context else config
        self.proposer = ProposerAgent(context)
        self.challenger = ChallengerAgent(context)
        self.moderator = ModeratorAgent(context)
        self.elo = EloStore(context=context)

    def conduct_debate(self, topic: Dict[str, Any]) -> DebateResult:
        title = topic.get("title", "<untitled>")
        logger.info("=== Debate start: %s ===", title)
        log_agent_action("HypothesisDebate", "start", {"topic": title})
        rounds: List[Dict[str, Any]] = []
        argument = self.proposer.build_argument(topic)
        logger.info("Round 1 proposer argument generated (%d chars)", len(argument))
        rebuttal = self.challenger.build_rebuttal(topic, argument)
        logger.info("Round 1 challenger rebuttal generated (%d chars)", len(rebuttal))
        objections = getattr(self.challenger, "_last_objections", []) or [
            {"criterion": "soundness", "objection": rebuttal[:500], "severity": 3, "status": "unresolved"}
        ]
        objections = _ensure_objections_are_dicts(objections)
        rounds.append({
            "round": 1,
            "proposer": argument,
            "challenger": rebuttal,
            "objections": objections,
        })

        min_r = self.runtime_config.debate_min_rounds
        max_r = self.runtime_config.debate_max_rounds
        current_objections = objections
        final = None

        # Skip additional rounds when challenger output was invalid — the
        # round-2 proposer response and challenger followup would be against
        # a garbled round-1, producing uninformative results.
        if getattr(self.challenger, "_challenger_invalid", False):
            logger.info("Challenger invalid in round 1 — skipping additional rounds")
            max_r = 1

        for r in range(2, max_r + 1):
            logger.info("Round %d: proposer responding to objections", r)
            response = self.proposer.respond_to_objections(topic, argument, current_objections)
            logger.info("Round %d: challenger evaluating followup objections", r)
            current_objections = self.challenger.followup_objections(topic, response, current_objections)
            current_objections = _ensure_objections_are_dicts(current_objections)
            unresolved_now_r = [o for o in current_objections if isinstance(o, dict) and o.get("status") != "resolved"]
            logger.info("Round %d complete: %d unresolved objections remaining", r, len(unresolved_now_r))
            rounds.append({
                "round": r,
                "proposer": response,
                "objections": current_objections,
            })
            argument = response
            tracker = get_tracker()
            if tracker:
                tracker.bump("debate_rounds")

            unresolved_now = [
                o for o in current_objections
                if isinstance(o, dict) and o.get("status") != "resolved"
            ]
            # Early stop after min rounds if no severe unresolved
            if r >= min_r and not any(_coerce_int(u.get("severity"), 0) >= 3 for u in unresolved_now):
                logger.info("Round %d: early stop — no severe unresolved objections", r)
                break

            # Ensemble disagreement can force another round
            if r >= min_r:
                mid = self.moderator.evaluate_debate(topic, rounds, current_objections)
                if not mid.get("needs_longer_debate"):
                    final = mid
                    logger.info("Round %d: moderator accepted result (no longer debate needed)", r)
                    break
                else:
                    logger.info("Round %d: moderator requests longer debate (disagreement=%.2f)", r, mid.get("disagreement", 0))
        else:
            final = None

        if not final:
            logger.info("Running final moderator evaluation after %d rounds", len(rounds))
            final = self.moderator.evaluate_debate(topic, rounds, current_objections)
            if final.get("needs_longer_debate") and len(rounds) < max_r:
                logger.info("Moderator still wants longer debate; running extra round")
                response = self.proposer.respond_to_objections(topic, argument, current_objections)
                current_objections = self.challenger.followup_objections(topic, response, current_objections)
                current_objections = _ensure_objections_are_dicts(current_objections)
                rounds.append({"round": len(rounds) + 1, "proposer": response, "objections": current_objections})
                final = self.moderator.evaluate_debate(topic, rounds, current_objections)

        if getattr(self.challenger, "_challenger_invalid", False):
            logger.warning("Challenger output was invalid; capping score and forcing FAIL")
            final = {
                **(final or {}),
                "score": min(float((final or {}).get("score", 5.0)), 4.0),
                "passed": False,
                "decision": "FAIL",
                "reasoning": "Challenger output invalid after retry; debate fails closed.",
            }

        unresolved_now = [
            o for o in current_objections
            if isinstance(o, dict) and o.get("status") != "resolved"
        ]
        # Skip Elo updates when debate failed on infrastructure (garbled
        # challenger) — the outcome is uninformative about hypothesis quality.
        if getattr(self.challenger, "_challenger_invalid", False):
            delta = 0.0
        else:
            delta = self.elo.update(topic.get("title", "general"), final["score"], final["passed"])
        # Feature 8: record seed-strategy outcome alongside kind-Elo
        # NOTE: outcome_status here is debate-derived (argument quality), NOT
        # experiment-derived. kind-Elo uses the same semantics — there is no
        # second Elo update at the post-experiment stage.
        seed_strategy = topic.get("seed_strategy")
        if seed_strategy:
            strategy_outcome = "debate_pass" if final["passed"] else (
                "debate_weak" if final["score"] >= 6 else "debate_fail"
            )
            self.elo.record_strategy_outcome(seed_strategy, strategy_outcome)
        unresolved_text = [f"[{u.get('criterion')}] {u.get('objection')}" for u in unresolved_now]

        result = DebateResult(
            topic=topic.get("title", ""),
            proposer_argument=rounds[0].get("proposer", ""),
            challenger_argument=rounds[0].get("challenger", ""),
            moderator_decision=final.get("decision", "FAIL"),
            score=float(final.get("score", 0)),
            passed=bool(final.get("passed")),
            reasoning=final.get("reasoning", ""),
            rounds=rounds,
            objections=current_objections,
            unresolved_objections=unresolved_text,
            ensemble_scores=final.get("ensemble_scores") or [],
            elo_delta=delta,
        )
        logger.info("=== Debate complete: %s | score=%.2f | decision=%s | rounds=%d | elo_delta=%.1f | unresolved=%d ===",
                     result.topic, result.score, result.moderator_decision,
                     len(rounds), result.elo_delta, len(unresolved_now))
        (self.context.memory if self.context else memory).add_debate_entry(
            result.topic,
            result.proposer_argument[:2000],
            result.challenger_argument[:2000],
            result.moderator_decision,
            result.score,
            structured_signal={
                "objection_type": (unresolved_now[0].get("criterion") if unresolved_now else "none"),
                "objection_types": sorted({str(item.get("criterion")) for item in unresolved_now if isinstance(item, dict) and item.get("criterion")}),
                "severity": max((_coerce_int(item.get("severity"), 0) for item in unresolved_now), default=0),
                "resolution_status": "unresolved" if unresolved_now else "resolved",
            },
            run_id=get_tracker().run_id if get_tracker() else "unknown",
            outcome_status="released",
        )
        tracker = get_tracker()
        if tracker:
            tracker.scratch("HypothesisDebate", "result", {
                "passed": result.passed,
                "score": result.score,
                "rounds": len(rounds),
                "unresolved": unresolved_text,
            })
        return result

    def revise_topic_from_objections(
        self,
        topic: Dict[str, Any],
        result: DebateResult,
    ) -> Optional[Dict[str, Any]]:
        """Produce one bounded, auditable scientific repair before rejection.

        The old workflow let a proposer answer objections rhetorically but did
        not permit it to change the experimental contract.  This method is the
        narrow repair loop: it can repair a baseline, metric, falsification
        condition, or MVE, but it may not erase provenance or silently turn a
        failed study into a pass.
        """
        if not getattr(self, "runtime_config", None) or getattr(getattr(self, "challenger", None), "_challenger_invalid", False):
            return None
        all_objections = _ensure_objections_are_dicts(result.objections or [])
        unresolved = [
            item for item in all_objections
            if isinstance(item, dict) and item.get("status") != "resolved"
        ]
        if not unresolved:
            return None
        repairable = {"baseline", "confounder", "evaluation", "statistical", "feasibility", "falsifiability", "soundness"}
        if not any(str(item.get("criterion")) in repairable for item in unresolved):
            return None
        logger.info("Attempting contract repair for '%s' (%d repairable objections)", topic.get("title", "<untitled>"), len(unresolved))
        current = _as_dict(topic.get("structured_hypothesis"))
        if not current:
            return None
        prompt = f"""
You are a research-methods repair agent. Revise the structured hypothesis below
to resolve the listed reviewer objections. Keep its research domain and every
evidence/provenance field intact. Do not claim novelty, performance, or a gap
without existing evidence. Produce a smaller executable study when necessary.

Current contract:
{json.dumps(current, indent=2, default=str)[:6000]}

Unresolved objections:
{json.dumps(unresolved, indent=2, default=str)[:3500]}

Hard requirements:
- preserve or strengthen the falsification condition;
- use only a catalogued local dataset or named synthetic data;
- name a baseline, metrics, a falsification test, and >=3 seeds;
- specify controls that address each confounder/baseline objection;
- return JSON for the complete structured hypothesis only.
"""
        for attempt in range(2):
            parsed = parse_json_from_llm(call_llm(prompt, temperature=0.2, tier="strong")) or {}
            admission = validate_topic_admission(parsed) if isinstance(parsed, dict) else {"admitted": False}
            if not admission.get("admitted"):
                log_agent_action("HypothesisDebate", "repair_contract_rejected", {
                    "attempt": attempt + 1,
                    "errors": admission.get("errors", ["invalid response"]),
                })
                continue
            revised = deepcopy(topic)
            revised["structured_hypothesis"] = parsed
            revised["falsifiable_prediction"] = parsed.get("falsification_condition", "")
            revised["research_question"] = parsed.get("research_question", "")
            revised.setdefault("repair_history", []).append({
                "source": "hypothesis_debate",
                "attempt": attempt + 1,
                "addressed_objections": [str(item.get("criterion")) for item in unresolved],
                "admission": admission,
            })
            log_agent_action("HypothesisDebate", "repair_contract_accepted", {
                "topic": topic.get("title"),
                "attempt": attempt + 1,
                "objections": len(unresolved),
            })
            return revised
        return None

    def conduct_with_repair(self, topic: Dict[str, Any]) -> List[DebateResult]:
        """Debate once, then allow exactly one contract repair and re-debate."""
        title = topic.get("title", "<untitled>")
        logger.info("conduct_with_repair: initial debate for '%s'", title)
        first = self.conduct_debate(topic)
        if first.passed:
            logger.info("conduct_with_repair: '%s' passed on first attempt", title)
            return [first]
        logger.info("conduct_with_repair: '%s' failed, attempting contract repair", title)
        revised = self.revise_topic_from_objections(topic, first)
        if not revised:
            logger.info("conduct_with_repair: repair failed for '%s', returning original result", title)
            return [first]
        # Preserve the candidate object's identity: the tournament/workflow
        # selects from its original topic list after this method returns.
        topic.clear()
        topic.update(revised)
        logger.info("conduct_with_repair: re-debating revised topic '%s'", revised.get("title", "<untitled>"))
        second = self.conduct_debate(revised)
        return [first, second]

    def conduct_tournament(self, topics: List[Dict[str, Any]], rounds: int = 1) -> List[DebateResult]:
        """Debate candidates until one passes, then hand off immediately.

        Each candidate is debated at most once. Remaining candidates are not
        debated once a hypothesis clears the bar — Planning should receive the
        accepted result without wasting LLM calls or re-debating failed topics.
        """
        logger.info("=== Tournament start: %d candidates ===", len(topics))
        results: List[DebateResult] = []
        debated_titles = set()
        for i, topic in enumerate(topics, 1):
            title = (topic.get("title") or "").strip()
            if not title or title.lower() in debated_titles:
                continue
            debated_titles.add(title.lower())
            logger.info("Tournament matchup %d/%d: '%s'", i, len(topics), title)
            attempts = self.conduct_with_repair(topic)
            results.extend(attempts)
            if attempts[-1].passed:
                logger.info("Tournament winner found: '%s' (score=%.2f)", title, attempts[-1].score)
                return sorted(results, key=lambda item: (item.passed, item.score), reverse=True)
        logger.info("Tournament complete: no candidate passed (%d total results)", len(results))
        return sorted(results, key=lambda item: (item.passed, item.score), reverse=True)
