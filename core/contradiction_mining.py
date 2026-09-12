"""Contradiction mining from paper abstracts."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.llm import call_llm
from core.utils import parse_json_from_llm

logger = logging.getLogger(__name__)


def find_contradictions(
    papers: List[Dict[str, Any]],
    max_papers: int = 15,
) -> List[Dict[str, Any]]:
    """Identify pairs of papers that make opposing empirical claims about the
    same method/metric/dataset.

    Fail-open: any failure → return [].
    """
    if not papers or len(papers) < 2:
        return []

    # Cap prompt size
    candidates = papers[:max_papers]

    abstracts_text = []
    for i, p in enumerate(candidates):
        title = p.get("title", "")[:150]
        abstract = p.get("abstract", "")[:500]
        if title or abstract:
            abstracts_text.append(f"[{i}] Title: {title}\nAbstract: {abstract}")

    if len(abstracts_text) < 2:
        return []

    prompt = f"""Identify pairs of papers among these abstracts that make opposing empirical claims
about the same method, metric, or dataset (e.g. one says X improves Y, another says X does
not improve Y or hurts it). Focus on genuine empirical disagreements, not just different topics.

Return JSON:
{{
  "contradictions": [
    {{
      "paper_a_title": "...",
      "paper_b_title": "...",
      "claim_a": "...",
      "claim_b": "...",
      "shared_subject": "..."
    }}
  ]
}}

Abstracts:
{json.dumps(abstracts_text, indent=2)[:8000]}"""

    try:
        raw = call_llm(prompt, temperature=0.3, tier="cheap")
        parsed = parse_json_from_llm(raw) or {}
        contradictions = parsed.get("contradictions") or []
    except Exception as e:
        logger.debug("find_contradictions: LLM call failed: %s", e)
        return []

    # Validate both titles actually match input papers (drop hallucinated pairs)
    input_titles = {p.get("title", "").strip().lower() for p in candidates if p.get("title")}
    validated = []
    for c in contradictions:
        if not isinstance(c, dict):
            continue
        pa = (c.get("paper_a_title") or "").strip().lower()
        pb = (c.get("paper_b_title") or "").strip().lower()
        if pa in input_titles and pb in input_titles and pa != pb:
            validated.append({
                "paper_a_title": c["paper_a_title"],
                "paper_b_title": c["paper_b_title"],
                "claim_a": str(c.get("claim_a", "")),
                "claim_b": str(c.get("claim_b", "")),
                "shared_subject": str(c.get("shared_subject", "")),
            })
    return validated
