"""
Semantic Scholar bulk search — free, no API key required for moderate volume,
higher effective throughput than arXiv's export API. Use as a primary text
search source alongside OpenAlex, with arXiv as a third/fallback leg rather
than the sole source, so an arXiv outage/429 storm doesn't stall discovery.
"""


from __future__ import annotations


import logging
from typing import Any, Dict, List, Optional


import requests


logger = logging.getLogger(__name__)


_S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"


def search_s2_bulk(
    query: str,
    headers: Dict[str, str],
    limit: int = 50,
    year_from: Optional[int] = None,
    fields: str = "title,abstract,year,externalIds,citationCount",
) -> List[Dict[str, Any]]:
    from core.api_gateway import get_gateway, RateLimitError

    def _do_fetch():
        params = {"query": query, "fields": fields, "limit": min(limit, 100)}
        if year_from:
            params["year"] = f"{year_from}-"
        r = requests.get(_S2_BULK_URL, headers=headers, params=params, timeout=30)
        if r.status_code == 429:
            raise RateLimitError("s2_bulk", float(r.headers.get("Retry-After", "3")))
        r.raise_for_status()
        return r.json().get("data", [])

    try:
        rows = get_gateway().request("s2_bulk", _do_fetch, retries=3, backoff_base=3.0)
    except Exception as e:
        logger.debug("S2 bulk search failed for %r: %s", query, e)
        return []

    results = []
    for row in rows or []:
        ext = row.get("externalIds") or {}
        results.append({
            "title": row.get("title", ""),
            "abstract": row.get("abstract", "") or "",
            "year": row.get("year", 0) or 0,
            "doi": ext.get("DOI"),
            "arxiv_id": ext.get("ArXiv"),
            "cited_by_count": row.get("citationCount", 0) or 0,
            "source": "s2_bulk",
        })
    return results
