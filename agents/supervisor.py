"""
SupervisorAgent — hard deterministic checks first, LLM soft review last.
Citation grounding + statistical validity gate soft scores.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from core.config import config
from core.utils import log_agent_action, validate_math_expression, verify_math_derivation, parse_json_from_llm, call_llm_json
from core.llm import call_llm
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.memory import memory
from core.verification import hard_verify_section, verify_citations, classify_claim
from core.run_log import get_tracker
from core.research_db import research_db
from core.contracts import ExperimentOutput, VerificationReport


REVIEW_CHECKLIST = (
    "every quantitative claim has a traceable artifact and uncertainty",
    "baselines and datasets match the committed experiment contract",
    "negative or inconclusive outcomes are reported",
    "the falsifiable prediction is explicitly evaluated and its verdict "
    "(supported / falsified / inconclusive) is stated with controls and robustness evidence",
    "limitations and reproducibility details are disclosed",
    "literature claims are supported by retrieved source text",
)


class SupervisorAgent:
    """Quality gate: hard checks dominate; LLM review is last filter only."""

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()
        self.ledger = self.context.research_db if self.context else research_db
        self.feedback_memory = self.context.memory if self.context else memory
        self.math_checker = MathChecker()
        self.code_checker = CodeChecker()
        self.reviewer_bot = ReviewerBot()

    @property
    def runtime_config(self):
        return self.context.config if self.context else config

    def evaluate_section(
        self,
        section_name: str,
        content: str,
        engineer_outputs: Optional[Dict[str, ExperimentOutput]] = None,
        content_requirements: Optional[str] = None,
    ) -> Tuple[float, str]:
        log_agent_action("SupervisorAgent", "start_evaluation", {"section": section_name})

        # --- HARD CHECKS FIRST ---
        hard = hard_verify_section(
            content,
            engineer_outputs=engineer_outputs,
            section_name=section_name,
            content_requirements=content_requirements,
        )
        self._record_claim_evidence(section_name, content, hard, engineer_outputs)
        math_score, math_fb = self.math_checker.evaluate(content, section_name)
        code_score, code_fb = self.code_checker.evaluate(content, section_name)

        hard_bundle_score = min(hard["score"], math_score, code_score)
        feedbacks = [
            f"HARD_CITATION_STATS: {hard['feedback']}",
            f"math_checker: {math_fb}",
            f"code_checker: {code_fb}",
        ]

        if not hard["passed"]:
            tracker = get_tracker()
            if tracker:
                tracker.bump("hard_check_fails")
            overall = min(hard_bundle_score, 4.0)
            overall_feedback = "HARD CHECK FAILED — deterministic checks block release.\n" + "\n".join(feedbacks)
            self.feedback_memory.add_feedback_entry(
                "SupervisorAgent",
                section_name,
                overall,
                overall_feedback,
                1,
                structured_signal={
                    "verdict": "block",
                    "score": float(overall),
                    "section": section_name,
                    "blocking": True,
                    "category": "hard_check_failed",
                },
                run_id=tracker.run_id if tracker else None,
                outcome_status="failed",
            )
            return overall, overall_feedback

        # --- SOFT LLM REVIEW LAST ---
        soft_score, soft_fb = self.reviewer_bot.evaluate(content, section_name)
        feedbacks.append(f"reviewer_bot: {soft_fb}")

        # Also run citation-aware hallucination soft check only after hard pass
        hall_score, hall_fb = self._soft_hallucination_check(content, section_name, hard)
        feedbacks.append(f"hallucination_checker: {hall_fb}")

        overall = (
            0.35 * hard["score"]
            + 0.15 * math_score
            + 0.15 * code_score
            + 0.20 * soft_score
            + 0.15 * hall_score
        )
        overall_feedback = "\n".join(feedbacks)

        tracker = get_tracker()
        self.feedback_memory.add_feedback_entry(
            "SupervisorAgent",
            section_name,
            overall,
            overall_feedback,
            1,
            structured_signal={
                "verdict": "pass" if overall >= self.runtime_config.supervisor_threshold else "revise",
                "score": float(overall),
                "section": section_name,
                "blocking": False,
                "category": "supervisor_review",
            },
            run_id=tracker.run_id if tracker else None,
            outcome_status="released" if overall >= self.runtime_config.supervisor_threshold else "revised",
        )
        log_agent_action("SupervisorAgent", "evaluation_complete", {
            "section": section_name,
            "overall_score": overall,
            "hard_passed": hard["passed"],
        })
        return overall, overall_feedback

    def _record_claim_evidence(self, section_name: str, content: str, hard: VerificationReport, engineer_outputs: Optional[Dict[str, ExperimentOutput]]):
        """Persist an auditable claim ledger instead of hiding evidence in reviewer prose."""
        tracker = get_tracker()
        if not tracker:
            return
        citation = hard.get("citation", {})
        for item in citation.get("resolved", []):
            identifier = item.get("doi") or item.get("arxiv_id") or "source"
            self.ledger.record_claim(tracker.run_id, section_name, identifier, "citation", "verified", item)
        for item in citation.get("failed", []):
            identifier = item.get("doi") or item.get("arxiv_id") or "source"
            self.ledger.record_claim(tracker.run_id, section_name, identifier, "citation", "failed", item)
        # Quantitative sentences are evidence claims; keep their deterministic check beside them.
        stat_status = "verified" if hard.get("passed") else "needs_revision"
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            if re.search(r"\d+(?:\.\d+)?\s*%|\b(?:mean|std|p\s*[<=>]|accuracy|f1)\b", sentence, re.I):
                claim_type = classify_claim(sentence, [item.get("experiment") for item in hard.get("statistics", [])])
                eligible = claim_type == "empirical_result" and bool(hard.get("statistics"))
                self.ledger.record_claim(tracker.run_id, section_name, sentence[:500], claim_type, stat_status if eligible else "unverifiable_by_construction", {"statistics": hard.get("statistics", [])})

    def _soft_hallucination_check(self, content: str, section_name: str, hard: Dict) -> Tuple[float, str]:
        prompt = f"""
After HARD citation resolution (passed={hard['passed']}), review remaining soft issues:
Section: {section_name}
Content: {content[:4000]}
Resolved citations: {json.dumps(hard.get('citation', {}).get('resolved', [])[:5], default=str)}

Score 1-10 for unsupported generalizations (not already caught by DOI checks).
JSON: {{"score": 8, "issues": [], "recommendations": []}}
"""
        try:
            result = call_llm_json(prompt, temperature=0.2, tier="judge", attempts=2, call_fn=call_llm) or {}
            return float(result.get("score", 5)), json.dumps(result)[:500]
        except Exception as e:
            return 5.0, str(e)


class MathChecker:
    def evaluate(self, content: str, section_name: str) -> Tuple[float, str]:
        expressions = self._extract_math_expressions(content)
        if not expressions:
            return 10.0, "No mathematical expressions found"

        valid = 0
        invalid = []
        derivation_notes = []
        for expr in expressions:
            if validate_math_expression(expr):
                valid += 1
            else:
                invalid.append(expr)

        # Attempt derivation pairs: "A = B" style
        for m in re.finditer(r"\$([^$]+)=([^$]+)\$", content):
            left, right = m.group(1).strip(), m.group(2).strip()
            if len(left) < 40 and len(right) < 40:
                check = verify_math_derivation([], left, right)
                derivation_notes.append(f"{left} == {right}? {check.get('passed')}")

        total = len(expressions)
        score = (valid / total) * 10 if total else 10.0
        fb = f"Math validation: {valid}/{total} valid"
        if invalid:
            fb += f" | invalid: {', '.join(invalid[:3])}"
        if derivation_notes:
            fb += " | derivations: " + "; ".join(derivation_notes[:3])
        return score, fb

    def _extract_math_expressions(self, content: str) -> List[str]:
        patterns = [
            r"\$([^$]+)\$",
            r"\\\[([^\]]+)\\\]",
            r"\\begin\{equation\}(.*?)\\end\{equation\}",
        ]
        expressions = []
        for pattern in patterns:
            expressions.extend(re.findall(pattern, content, re.DOTALL))
        return list(set(expressions))


class CodeChecker:
    def evaluate(self, content: str, section_name: str) -> Tuple[float, str]:
        snippets = self._extract_code_snippets(content)
        if not snippets:
            return 10.0, "No code snippets found"
        valid = 0
        invalid = []
        for snippet in snippets:
            if self._validate_code_snippet(snippet):
                valid += 1
            else:
                invalid.append(snippet[:50] + "...")
        total = len(snippets)
        score = (valid / total) * 10 if total else 10.0
        fb = f"Code validation: {valid}/{total} snippets valid"
        if invalid:
            fb += f" | invalid: {', '.join(invalid[:3])}"
        return score, fb

    def _extract_code_snippets(self, content: str) -> List[str]:
        patterns = [
            r"```[\w]*\n(.*?)\n```",
            r"`([^`]+)`",
            r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}",
        ]
        snippets = []
        for pattern in patterns:
            snippets.extend(re.findall(pattern, content, re.DOTALL))
        return list(set(snippets))

    def _validate_code_snippet(self, snippet: str) -> bool:
        try:
            if any(k in snippet for k in ("def ", "import ", "class ")):
                compile(snippet, "<string>", "exec")
                return True
            return self._balanced(snippet)
        except Exception:
            return False

    def _balanced(self, text: str) -> bool:
        stack = []
        pairs = {")": "(", "}": "{", "]": "["}
        for char in text:
            if char in "({[":
                stack.append(char)
            elif char in ")}]":
                if not stack or stack.pop() != pairs[char]:
                    return False
        return len(stack) == 0


class ReviewerBot:
    """LLM soft peer review — only meaningful AFTER hard checks."""

    def evaluate(self, content: str, section_name: str) -> Tuple[float, str]:
        prompt = f"""
You are a peer reviewer. Soft qualitative review ONLY (hard citation/stats already checked).
Section: {section_name}
Content: {content[:5000]}

Checklist:
{json.dumps(REVIEW_CHECKLIST)}

Criteria 1-10: clarity, accuracy, flow, completeness, writing.
JSON:
{{"scores": {{"clarity": 8, "accuracy": 7, "flow": 8, "completeness": 6, "writing": 7}},
    "overall_score": 7.2, "checklist": {{"item": true}}, "strengths": [], "weaknesses": [], "suggestions": []}}
"""
        try:
            result = call_llm_json(prompt, temperature=0.3, tier="judge", attempts=2, call_fn=call_llm) or {}
            score = float(result.get("overall_score", 5.0))
            fb = f"Peer review score: {score}/10"
            if result.get("weaknesses"):
                fb += f" | weaknesses: {', '.join(result['weaknesses'][:2])}"
            return score, fb
        except Exception as e:
            return 5.0, f"Peer review error: {e}"
