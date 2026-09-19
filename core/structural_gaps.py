"""Structural gap mining via bibliographic coupling analysis."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def find_coupling_gaps(
    papers: List[Dict[str, Any]],
    s2_headers: Dict[str, str],
    base_url: str = "https://api.semanticscholar.org/graph/v1",
    min_shared_refs: int = 3,
    max_pairs: int = 8,
) -> List[Dict[str, Any]]:
    """Find bibliographic coupling gaps: pairs of papers that share many
    references but do not cite each other directly.

    All S2 calls route through the API gateway (pacing + breaker + adaptive
    rate). Fail-open: any API error or <2 resolvable papers → return [].
    """
    if not papers or len(papers) < 2:
        return []

    from .api_gateway import get_gateway, RateLimitError

    def _s2_get(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
        """Single gateway-managed attempt; gateway owns retries and pacing."""

        def _do_fetch():
            r = requests.get(url, headers=s2_headers, params=params or {}, timeout=15)
            if r.status_code == 429:
                raise RateLimitError("s2", float(r.headers.get("Retry-After", "3")))
            return r

        try:
            if not get_gateway().is_available("s2"):
                return None
            return get_gateway().request("s2", _do_fetch, retries=1, backoff_base=2.0)
        except Exception as e:
            logger.debug("find_coupling_gaps: S2 fetch error for %s: %s", url, e)
            return None

    candidate_papers = papers[:20]
    paper_refs: Dict[str, List[str]] = {}
    paper_titles: Dict[str, str] = {}

    # Skip the whole loop when the S2 breaker is open — nothing will resolve.
    try:
        if not get_gateway().is_available("s2"):
            return []
    except Exception:
        pass

    for p in candidate_papers:
        raw_doi = (p.get("doi") or "").replace("https://doi.org/", "").strip()
        raw_arxiv = (p.get("arxiv_id") or "").strip()
        raw_s2 = (p.get("s2_paper_id") or "").strip()
        if raw_s2:
            paper_id = raw_s2
        elif raw_doi:
            paper_id = f"DOI:{raw_doi}"
        elif raw_arxiv:
            arxiv_num = raw_arxiv.split("/")[-1] if "/" in raw_arxiv else raw_arxiv
            paper_id = f"ARXIV:{arxiv_num}"
        else:
            continue
        title = p.get("title", "unknown")
        paper_titles[paper_id] = title
        url = f"{base_url}/paper/{paper_id}"
        r = _s2_get(url, params={"fields": "references.paperId,citations.paperId"})
        if r is None or r.status_code != 200:
            paper_refs[paper_id] = []
            continue
        try:
            data = r.json()
            ref_ids = set()
            for ref in data.get("references") or []:
                rid = ref.get("paperId")
                if rid:
                    ref_ids.add(rid)
            paper_refs[paper_id] = list(ref_ids)
            paper_titles[paper_id] = data.get("title") or title
        except Exception as e:
            logger.debug("find_coupling_gaps: S2 parse error for %s: %s", paper_id, e)
            paper_refs[paper_id] = []

    resolved_ids = [pid for pid, refs in paper_refs.items() if refs]
    if len(resolved_ids) < 2:
        return []

    # Precompute citation sets for direct-citation check
    citation_sets: Dict[str, set] = {}
    for pid in resolved_ids:
        citation_sets[pid] = set(paper_refs.get(pid, []))

    pairs: List[Dict[str, Any]] = []
    for i in range(len(resolved_ids)):
        for j in range(i + 1, len(resolved_ids)):
            a_id = resolved_ids[i]
            b_id = resolved_ids[j]
            refs_a = citation_sets.get(a_id, set())
            refs_b = citation_sets.get(b_id, set())
            shared = refs_a & refs_b
            if len(shared) < min_shared_refs:
                continue
            # No direct citation check
            if b_id in refs_a or a_id in refs_b:
                continue
            pairs.append({
                "paper_a": paper_titles.get(a_id, a_id),
                "paper_b": paper_titles.get(b_id, b_id),
                "paper_a_id": a_id,
                "paper_b_id": b_id,
                "shared_reference_count": len(shared),
                "gap_type": "bibliographic_coupling_no_direct_citation",
            })

    pairs.sort(key=lambda x: x["shared_reference_count"], reverse=True)
    return pairs[:max_pairs]
