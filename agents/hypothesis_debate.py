"""
Hypothesis Debate — multi-round adversarial debate with ensemble judging + Elo.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import call_llm, log_agent_action, parse_json_from_llm
from core.llm import get_llm_client
from core.memory import memory
from core.run_log import get_tracker


REVIEWER_CHECKLIST = [
    "soundness",
    "significance",
    "reproducibility",
    "ethics",
    "novelty",
    "feasibility",
]


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
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or config.elo_ratings_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ratings: Dict[str, float] = {}
        if self.path.exists():
            try:
                self.ratings = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.ratings = {}

    def get(self, key: str, default: float = 1500.0) -> float:
        return float(self.ratings.get(key, default))

    def update(self, hypothesis_key: str, score: float, passed: bool) -> float:
        """Simple Elo vs a fixed 'reviewer bar' opponent at 1500."""
        ra = self.get(hypothesis_kind(hypothesis_key))
        rb = 1500.0
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        outcome = 1.0 if passed else (0.5 if score >= 6 else 0.0)
        k = 32
        delta = k * (outcome - ea)
        new = ra + delta
        self.ratings[hypothesis_kind(hypothesis_key)] = new
        self.path.write_text(json.dumps(self.ratings, indent=2), encoding="utf-8")
        return delta


def hypothesis_kind(title: str) -> str:
    """Coarse bucket for Elo (kinds of hypotheses)."""
    t = title.lower()
    for kind in ("attention", "graph", "diffusion", "reinforcement", "federated", "llm", "vision", "nlp"):
        if kind in t:
            return kind
    return "general"


class ProposerAgent:
    def __init__(self):
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
    def __init__(self):
        self.client = get_llm_client()

    def build_rebuttal(self, topic: Dict[str, Any], proposer_argument: str) -> str:
        """Checklist-grounded critique — must address each reviewer criterion."""
        rating = float(topic.get("elo_rating", EloStore().get(hypothesis_kind(topic.get("title", "")))))
        kind = topic.get("hypothesis_kind") or hypothesis_kind(topic.get("title", ""))
        history_note = (
            f"This is a {kind} hypothesis with a low historical Elo ({rating:.0f}). "
            "Scrutinize the failure modes that have historically made this kind fail."
            if rating < 1450 else
            f"This is a {kind} hypothesis with historical Elo {rating:.0f}; still independently verify its claims."
        )
        prompt = f"""
You are a critical reviewer. Address EACH criterion explicitly:
{', '.join(REVIEWER_CHECKLIST)}

Topic: {topic.get('title')}
Argument: {proposer_argument[:4000]}
Historical signal: {history_note}

Return JSON:
{{
  "objections": [
    {{"criterion": "soundness", "objection": "...", "severity": 1-5}}
  ],
  "summary_rebuttal": "..."
}}
Require at least one objection per criterion (can note 'no major issue' with severity 1).
"""
        raw = call_llm(prompt, temperature=0.5, tier="strong")
        parsed = parse_json_from_llm(raw) or {}
        objections = parsed.get("objections") or []
        summary = parsed.get("summary_rebuttal") or raw
        # stash on instance for multi-round
        self._last_objections = objections
        log_agent_action("ChallengerAgent", "built_rebuttal", {"n_objections": len(objections)})
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
        parsed = parse_json_from_llm(call_llm(prompt, temperature=0.3, tier="judge")) or {}
        return parsed.get("unresolved") or []


class ModeratorAgent:
    def __init__(self):
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
Pass threshold: {config.debate_pass_threshold}

Score overall scientific merit 1-10. PASS only if score >= {config.debate_pass_threshold}
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
        passed = agreed and mean >= config.debate_pass_threshold and not any(
            (u.get("severity") or 0) >= 4 for u in unresolved
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
        models = config.get_ensemble_models()
        if models:
            return models[:3]
        # Single configured judge/strong model repeated is weak; use cheap+strong if set
        out = []
        for tier in ("judge", "strong", "cheap"):
            m = config.resolve_model(tier)
            if m and m not in out:
                out.append(m)
        return out[:3] or [config.resolve_model("default")]


class HypothesisDebateSystem:
    def __init__(self):
        self.proposer = ProposerAgent()
        self.challenger = ChallengerAgent()
        self.moderator = ModeratorAgent()
        self.elo = EloStore()

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

        min_r = config.debate_min_rounds
        max_r = config.debate_max_rounds
        unresolved = objections

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
            if r >= min_r and not any((u.get("severity") or 0) >= 3 for u in unresolved):
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
        memory.add_debate_entry(
            result.topic,
            result.proposer_argument[:2000],
            result.challenger_argument[:2000],
            result.moderator_decision,
            result.score,
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
