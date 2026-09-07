"""
Hypothesis Debate — multi-round adversarial debate with ensemble judging + Elo.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import log_agent_action, parse_json_from_llm, is_degenerate_llm_output
from core.llm import call_llm
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.memory import memory
from core.run_log import get_tracker
from core.capabilities import SANDBOX_CAPABILITY_MANIFEST, check_plan_feasibility


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
            except Exception:
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
        return delta


def hypothesis_kind(title: str) -> str:
    """Coarse bucket for Elo (kinds of hypotheses)."""
    t = title.lower()
    for kind in ("attention", "graph", "diffusion", "reinforcement", "federated", "llm", "vision", "nlp"):
        if kind in t:
            return kind
    return "general"


class ProposerAgent:
    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()

    def build_argument(self, topic: Dict[str, Any]) -> str:
        prompt = f"""
You are a research proposer. Build a compelling, realistic argument.
Topic: {topic.get('title')}
Description: {topic.get('description')}
Rationale: {topic.get('rationale', 'N/A')}
Impact: {topic.get('impact', 'N/A')}
Feasibility: {topic.get('feasibility', 5)}/10

Include: hypothesis, theory, evidence, novelty, methodology, expected outcomes, falsifiable prediction.
"""
        arg = call_llm(prompt, temperature=0.7, tier="strong")
        log_agent_action("ProposerAgent", "built_argument", {"len": len(arg or "")})
        return arg or "Failed to build argument."

    def respond_to_objections(
        self,
        topic: Dict[str, Any],
        prior_argument: str,
        objections: List[Dict[str, Any]],
    ) -> str:
        """Respond to EACH specific objection (not a global rebuttal)."""
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
        return call_llm(prompt, temperature=0.5, tier="strong") or ""


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
        """Checklist-grounded critique — must address each reviewer criterion."""
        rating = float(topic.get("elo_rating", EloStore(context=self.context).get(hypothesis_kind(topic.get("title", "")))))
        kind = topic.get("hypothesis_kind") or hypothesis_kind(topic.get("title", ""))
        history_note = (
            f"This is a {kind} hypothesis with a low historical Elo ({rating:.0f}). "
            "Scrutinize the failure modes that have historically made this kind fail."
            if rating < 1450 else
            f"This is a {kind} hypothesis with historical Elo {rating:.0f}; still independently verify its claims."
        )
        prior_tags = self._prior_objection_tags()
        prompt = f"""
You are a critical reviewer. Address EACH criterion explicitly:
{', '.join(REVIEWER_CHECKLIST)}

Topic: {topic.get('title')}
Argument: {proposer_argument[:4000]}
Historical signal: {history_note}
Prior objection tags only (never reuse prior prose): {json.dumps(prior_tags)[:1500]}
Sandbox capability manifest: {json.dumps(SANDBOX_CAPABILITY_MANIFEST.as_dict(), sort_keys=True)}

Treat any requirement outside this manifest as a first-class feasibility objection
with severity 5. Do not assume unavailable network, GPU, libraries, or data.

Return JSON:
{{
  "objections": [
    {{"criterion": "soundness", "objection": "...", "severity": 1-5}}
  ],
  "summary_rebuttal": "..."
}}
Require at least one objection per criterion (can note 'no major issue' with severity 1).
"""
        self._challenger_invalid = False
        parsed: Dict[str, Any] = {}
        objections: List[Dict[str, Any]] = []
        summary = ""
        raw = ""
        for attempt in range(2):
            raw = call_llm(prompt, temperature=0.5, tier="strong")
            parsed = parse_json_from_llm(raw) or {}
            objections = list(parsed.get("objections") or [])
            summary = parsed.get("summary_rebuttal") or raw
            if isinstance(summary, str) and not is_degenerate_llm_output(summary, min_chars=40) and len(objections) > 0:
                break
            log_agent_action("ChallengerAgent", "degenerate_rebuttal", {
                "attempt": attempt + 1,
                "n_objections": len(objections),
                "preview": str(summary)[:120],
            })
        else:
            # Fail closed: never treat empty/garbled challenger output as zero objections.
            self._challenger_invalid = True
            objections = [{
                "criterion": "soundness",
                "objection": "Challenger produced empty or garbled output; debate fails closed.",
                "severity": 5,
                "source": "challenger_output_guard",
            }]
            summary = "CHALLENGER_OUTPUT_INVALID"
            log_agent_action("ChallengerAgent", "rebuttal_invalid_after_retry", {"raw_preview": str(raw)[:200]})

        feasibility_errors = check_plan_feasibility(
            {"methodology": topic.get("description", ""), "experiments": [{"dataset": {"name": topic.get("dataset_plan", "")}}]},
            SANDBOX_CAPABILITY_MANIFEST,
        )
        objections.extend({
            "criterion": "feasibility",
            "objection": reason,
            "severity": 5,
            "source": "capability_manifest",
        } for reason in feasibility_errors)
        self._last_objections = objections
        log_agent_action("ChallengerAgent", "built_rebuttal", {
            "n_objections": len(objections),
            "invalid": bool(self._challenger_invalid),
        })
        return summary if isinstance(summary, str) else json.dumps(parsed)

    def followup_objections(
        self,
        topic: Dict[str, Any],
        proposer_response: str,
        prior_objections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prompt = f"""
Given the proposer's point-by-point responses, which objections remain unresolved?
Prior objections: {json.dumps(prior_objections)[:3000]}
Proposer responses: {proposer_response[:3000]}

Return JSON: {{"unresolved": [{{"criterion": "...", "objection": "...", "severity": 3}}], "resolved": ["..."]}}
"""
        for attempt in range(2):
            raw = call_llm(prompt, temperature=0.3, tier="judge")
            if is_degenerate_llm_output(raw, min_chars=20):
                log_agent_action("ChallengerAgent", "degenerate_followup", {"attempt": attempt + 1})
                continue
            parsed = parse_json_from_llm(raw) or {}
            if "unresolved" in parsed or "resolved" in parsed:
                return parsed.get("unresolved") or []
        # Fail closed: keep prior objections rather than inventing a clean slate.
        return list(prior_objections)


class ModeratorAgent:
    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.runtime_config = self.context.config if self.context else config
        self.client = get_llm_client()

    def evaluate_debate(
        self,
        topic: Dict[str, Any],
        rounds: List[Dict[str, Any]],
        unresolved: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Ensemble judge: multiple models; disagreement extends debate signal."""
        models = self._judge_models()
        scores = []
        reasonings = []
        transcript = json.dumps(rounds, default=str)[:6000]
        base_prompt = f"""
Moderate this multi-round research debate.
Topic: {topic.get('title')}
Transcript: {transcript}
Unresolved objections: {json.dumps(unresolved)[:2000]}
Pass threshold: {self.runtime_config.debate_pass_threshold}

Score overall scientific merit 1-10. PASS only if score >= {self.runtime_config.debate_pass_threshold}
and unresolved high-severity (>=4) objections are empty.

JSON: {{"score": 7.5, "passed": true, "reasoning": "...", "decision": "PASS|FAIL"}}
"""
        for model in models:
            raw = call_llm(base_prompt, temperature=0.2, tier="judge", model=model)
            parsed = parse_json_from_llm(raw) or {}
            try:
                scores.append(float(parsed.get("score", 5)))
            except (TypeError, ValueError):
                scores.append(5.0)
            reasonings.append(parsed.get("reasoning", raw[:300]))

        if not scores:
            scores = [5.0]
            reasonings = ["no judge response"]

        mean = sum(scores) / len(scores)
        disagreement = max(scores) - min(scores) if len(scores) > 1 else 0.0
        # Agreement required: if disagreement high, do not pass even if mean is high
        agreed = disagreement <= 1.5
        passed = agreed and mean >= self.runtime_config.debate_pass_threshold and not any(
            _coerce_int(u.get("severity"), 0) >= 4 for u in unresolved
        )
        return {
            "score": mean,
            "passed": passed,
            "ensemble_scores": scores,
            "disagreement": disagreement,
            "needs_longer_debate": disagreement > 1.5,
            "reasoning": " | ".join(reasonings[:3]),
            "decision": "PASS" if passed else "FAIL",
        }

    def _judge_models(self) -> List[str]:
        models = self.runtime_config.get_ensemble_models()
        if models:
            return models[:3]
        # Single configured judge/strong model repeated is weak; use cheap+strong if set
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
        log_agent_action("HypothesisDebate", "start", {"topic": topic.get("title")})
        rounds: List[Dict[str, Any]] = []
        argument = self.proposer.build_argument(topic)
        rebuttal = self.challenger.build_rebuttal(topic, argument)
        objections = getattr(self.challenger, "_last_objections", []) or [
            {"criterion": "soundness", "objection": rebuttal[:500], "severity": 3}
        ]
        rounds.append({
            "round": 1,
            "proposer": argument,
            "challenger": rebuttal,
            "objections": objections,
        })

        min_r = self.runtime_config.debate_min_rounds
        max_r = self.runtime_config.debate_max_rounds
        unresolved = objections
        final = None

        for r in range(2, max_r + 1):
            response = self.proposer.respond_to_objections(topic, argument, unresolved)
            unresolved = self.challenger.followup_objections(topic, response, unresolved)
            rounds.append({
                "round": r,
                "proposer": response,
                "objections": unresolved,
            })
            argument = response
            tracker = get_tracker()
            if tracker:
                tracker.bump("debate_rounds")

            # Early stop after min rounds if no severe unresolved
            if r >= min_r and not any(_coerce_int(u.get("severity"), 0) >= 3 for u in unresolved):
                break

            # Ensemble disagreement can force another round
            if r >= min_r:
                mid = self.moderator.evaluate_debate(topic, rounds, unresolved)
                if not mid.get("needs_longer_debate"):
                    # continue to final eval below with this mid result cached
                    final = mid
                    break
        else:
            final = None

        if not final:
            final = self.moderator.evaluate_debate(topic, rounds, unresolved)
            # If disagreement, one forced extra round already handled in loop; mark longer
            if final.get("needs_longer_debate") and len(rounds) < max_r:
                response = self.proposer.respond_to_objections(topic, argument, unresolved)
                unresolved = self.challenger.followup_objections(topic, response, unresolved)
                rounds.append({"round": len(rounds) + 1, "proposer": response, "objections": unresolved})
                final = self.moderator.evaluate_debate(topic, rounds, unresolved)

        if getattr(self.challenger, "_challenger_invalid", False):
            final = {
                **(final or {}),
                "score": min(float((final or {}).get("score", 5.0)), 4.0),
                "passed": False,
                "decision": "FAIL",
                "reasoning": "Challenger output invalid after retry; debate fails closed.",
            }

        delta = self.elo.update(topic.get("title", "general"), final["score"], final["passed"])
        unresolved_text = [f"[{u.get('criterion')}] {u.get('objection')}" for u in unresolved]

        result = DebateResult(
            topic=topic.get("title", ""),
            proposer_argument=rounds[0].get("proposer", ""),
            challenger_argument=rounds[0].get("challenger", ""),
            moderator_decision=final.get("decision", "FAIL"),
            score=float(final.get("score", 0)),
            passed=bool(final.get("passed")),
            reasoning=final.get("reasoning", ""),
            rounds=rounds,
            objections=objections,
            unresolved_objections=unresolved_text,
            ensemble_scores=final.get("ensemble_scores") or [],
            elo_delta=delta,
        )
        (self.context.memory if self.context else memory).add_debate_entry(
            result.topic,
            result.proposer_argument[:2000],
            result.challenger_argument[:2000],
            result.moderator_decision,
            result.score,
            structured_signal={
                "objection_type": (unresolved[0].get("criterion") if unresolved else "none"),
                "objection_types": sorted({str(item.get("criterion")) for item in unresolved if item.get("criterion")}),
                "severity": max((_coerce_int(item.get("severity"), 0) for item in unresolved), default=0),
                "resolution_status": "unresolved" if unresolved else "resolved",
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

    def conduct_tournament(self, topics: List[Dict[str, Any]], rounds: int = 1) -> List[DebateResult]:
        """Debate candidates until one passes, then hand off immediately.

        Each candidate is debated at most once. Remaining candidates are not
        debated once a hypothesis clears the bar — Planning should receive the
        accepted result without wasting LLM calls or re-debating failed topics.
        """
        results: List[DebateResult] = []
        debated_titles = set()
        for topic in topics:
            title = (topic.get("title") or "").strip()
            if not title or title.lower() in debated_titles:
                continue
            debated_titles.add(title.lower())
            result = self.conduct_debate(topic)
            results.append(result)
            if result.passed:
                return sorted(results, key=lambda item: (item.passed, item.score), reverse=True)
        return sorted(results, key=lambda item: (item.passed, item.score), reverse=True)
