"""Auditable cross-paper synthesis for research discovery.

The topic hunter must not infer a research gap from a single attractive paper
or ask an LLM to manufacture citations after it has invented an idea.  This
module builds a small, deterministic evidence graph from the retrieved corpus.
It proposes *bridges* between papers with complementary method and application
signals.  A bridge is evidence for a candidate to screen, never proof that a
research gap exists.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from typing import Any, Dict, Iterable, List

from .datasets import list_datasets


_WORD = re.compile(r"[a-z][a-z0-9_-]{2,}")
_STOPWORDS = {
    "about", "after", "also", "among", "analysis", "approach", "based", "between",
    "both", "data", "different", "during", "each", "for", "from", "into", "method",
    "methods", "model", "models", "more", "new", "our", "paper", "results", "show",
    "study", "that", "their", "these", "this", "using", "with", "work",
}

# Kept deliberately compact and transparent.  These are research roles, not a
# claim that two terms are scientifically interchangeable.
_ROLE_TERMS = {
    "method": {"algorithm", "benchmark", "classifier", "estimation", "learning", "modeling", "optimization", "prediction", "simulation", "validation"},
    "evaluation": {"accuracy", "bias", "calibration", "evaluation", "fairness", "metric", "performance", "reproducibility", "robustness"},
    "setting": {"clinical", "distributed", "education", "energy", "healthcare", "iot", "manufacturing", "network", "scientific", "security"},
}


def _tokens(text: str) -> List[str]:
    return [word for word in _WORD.findall((text or "").lower()) if word not in _STOPWORDS]


def _snippet(text: str, term: str, limit: int = 240) -> str:
    """Return an exact source excerpt around a term, for reviewer inspection."""
    text = (text or "").strip()
    index = text.lower().find(term.lower())
    if index < 0:
        return text[:limit]
    start = max(0, index - limit // 3)
    end = min(len(text), start + limit)
    return text[start:end]


def _roles(tokens: Iterable[str]) -> Dict[str, List[str]]:
    present = set(tokens)
    return {role: sorted(present & vocabulary) for role, vocabulary in _ROLE_TERMS.items()}


def build_cross_paper_evidence_map(
    papers: Iterable[Dict[str, Any]],
    *,
    max_papers: int = 20,
    max_bridges: int = 12,
) -> Dict[str, Any]:
    """Build a provenance-preserving corpus map and conservative bridge list.

    A bridge requires one paper to contribute a method/evaluation signal and a
    different paper to contribute an application setting.  Every returned
    bridge carries paper ids and verbatim excerpts; consumers can therefore
    reject unsupported synthesis before the debate stage.
    """
    nodes: List[Dict[str, Any]] = []
    for index, paper in enumerate(list(papers)[:max_papers]):
        if not isinstance(paper, dict):
            continue
        title = str(paper.get("title") or "").strip()
        abstract = str(paper.get("abstract") or "").strip()
        if not title or not abstract:
            continue
        tokens = _tokens(f"{title} {abstract}")
        counts = Counter(tokens)
        salient = [term for term, _ in counts.most_common(20)]
        node_id = str(paper.get("id") or paper.get("doi") or paper.get("arxiv_id") or f"paper-{index}")
        nodes.append({
            "paper_id": node_id,
            "title": title,
            "source": paper.get("source") or "retrieved_corpus",
            "terms": salient,
            "roles": _roles(tokens),
            "abstract": abstract,
        })

    bridges: List[Dict[str, Any]] = []
    for left, right in combinations(nodes, 2):
        left_methods = left["roles"]["method"] + left["roles"]["evaluation"]
        right_methods = right["roles"]["method"] + right["roles"]["evaluation"]
        left_settings = left["roles"]["setting"]
        right_settings = right["roles"]["setting"]

        # Directional transfer hypotheses.  Do not emit a bridge without both
        # a transferable mechanism and a materially distinct target setting.
        directions = ((left, right, left_methods, right_settings), (right, left, right_methods, left_settings))
        for method_paper, setting_paper, mechanisms, settings in directions:
            if not mechanisms or not settings:
                continue
            shared = set(method_paper["terms"]) & set(setting_paper["terms"])
            # Reject near duplicates.  Cross-paper synthesis needs meaningful
            # complementarity, not two papers that merely repeat one topic.
            union = set(method_paper["terms"]) | set(setting_paper["terms"])
            lexical_overlap = len(shared) / max(len(union), 1)
            if lexical_overlap > 0.45:
                continue
            mechanism = mechanisms[0]
            setting = settings[0]
            bridges.append({
                "bridge_id": f"bridge-{method_paper['paper_id']}-{setting_paper['paper_id']}-{mechanism}-{setting}",
                "bridge_type": "method_to_setting_transfer",
                "source_paper_ids": [method_paper["paper_id"], setting_paper["paper_id"]],
                "method_signal": mechanism,
                "target_setting_signal": setting,
                "shared_terms": sorted(shared)[:8],
                "complementarity": round(1.0 - lexical_overlap, 3),
                "evidence": [
                    {"paper_id": method_paper["paper_id"], "title": method_paper["title"], "excerpt": _snippet(method_paper["abstract"], mechanism)},
                    {"paper_id": setting_paper["paper_id"], "title": setting_paper["title"], "excerpt": _snippet(setting_paper["abstract"], setting)},
                ],
                "screening_question": f"Can {mechanism} be evaluated or adapted for {setting} under a controlled local protocol?",
                "status": "candidate_requires_literature_screening",
            })

    bridges.sort(key=lambda item: (-item["complementarity"], item["method_signal"], item["target_setting_signal"]))
    # Abstracts are intentionally excluded from the external representation;
    # exact excerpts are sufficient for prompts and keep context bounded.
    public_nodes = [{key: value for key, value in node.items() if key != "abstract"} for node in nodes]
    return {
        "schema_version": "cross-paper-evidence-map/v1",
        "papers": public_nodes,
        "bridges": bridges[:max_bridges],
        "limitations": [
            "Bridges express evidence-backed candidates, not verified novelty or causal claims.",
            "Every candidate requires the ordinary literature, novelty, feasibility, and debate gates.",
        ],
    }


def validate_candidate_bridge_claim(candidate: Dict[str, Any], evidence_map: Dict[str, Any]) -> Dict[str, Any]:
    """Require a synthesis-backed candidate to identify its supporting bridges.

    This is an anti-hallucination gate: a model cannot claim that two papers
    imply a research opportunity unless it points to bridge IDs created from
    the retrieved corpus.  It validates provenance, not scientific truth.
    """
    available = {str(item.get("bridge_id")): item for item in evidence_map.get("bridges", [])}
    requested = candidate.get("evidence_bridge_ids") or []
    if isinstance(requested, str):
        requested = [requested]
    requested = [str(value) for value in requested]
    if not available:
        return {"valid": True, "bridge_ids": [], "reason": "no_cross_paper_bridges_available"}
    if not requested:
        return {"valid": False, "bridge_ids": [], "reason": "candidate_did_not_cite_cross_paper_evidence"}
    unknown = [value for value in requested if value not in available]
    if unknown:
        return {"valid": False, "bridge_ids": requested, "reason": f"unknown_evidence_bridge_ids: {unknown}"}
    text = " ".join(str(candidate.get(key) or "") for key in ("title", "description", "rationale", "contribution")).lower()
    supported = []
    for bridge_id in requested:
        bridge = available[bridge_id]
        terms = {str(bridge.get("method_signal", "")).lower(), str(bridge.get("target_setting_signal", "")).lower()}
        if any(term and term in text for term in terms):
            supported.append(bridge_id)
    if not supported:
        return {"valid": False, "bridge_ids": requested, "reason": "candidate_text_does_not_match_cited_bridge_signals"}
    return {"valid": True, "bridge_ids": supported, "reason": "grounded_cross_paper_bridge"}


def validate_topic_admission(structured_hypothesis: Dict[str, Any]) -> Dict[str, Any]:
    """Apply deterministic, pre-debate admission checks to a proposed study.

    This prevents the expensive debate phase from receiving an attractive
    narrative with no executable experiment.  It is intentionally narrower
    than scientific review: passing means "ready for debate", never "true".
    """
    contract = structured_hypothesis or {}
    errors: List[str] = []
    for field in ("research_question", "hypothesis", "dependent_variables", "falsification_condition"):
        if not contract.get(field):
            errors.append(f"missing required hypothesis field: {field}")

    mve = contract.get("minimum_viable_experiment")
    if not isinstance(mve, dict):
        errors.append("missing minimum_viable_experiment")
    else:
        dataset = str(mve.get("dataset") or "").strip()
        catalog = {item["name"] for item in list_datasets()}
        if not dataset:
            errors.append("MVE does not name a dataset")
        elif dataset not in catalog and "synthetic" not in dataset.lower():
            errors.append(f"MVE dataset is not locally available: {dataset}")
        if not (mve.get("baseline") or mve.get("baselines")):
            errors.append("MVE lacks a named baseline")
        if not mve.get("metrics"):
            errors.append("MVE lacks evaluation metrics")
        if not mve.get("falsification_test"):
            errors.append("MVE lacks a falsification test")
        try:
            seeds = int(mve.get("seeds", 0))
        except (TypeError, ValueError):
            seeds = 0
        if seeds < 3:
            errors.append("MVE needs at least three independent seeds")

    return {
        "admitted": not errors,
        "errors": errors,
        "contract_version": "topic-admission/v1",
    }
