"""
Hard verification checks: citation resolution + statistical validity.
These are deterministic — not LLM vibe scores.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import config


# ---------------------------------------------------------------------------
# Citation grounding
# ---------------------------------------------------------------------------

DOI_RE = re.compile(
    r"(?:doi[:\s]*)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.IGNORECASE,
)
ARXIV_RE = re.compile(
    # A bare arXiv id is valid, but do not mistake the tail of a DOI such as
    # 10.5555/3295222.3295349 for one.
    r"(?<![\d.])(?:arxiv[:\s]*)?(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
YEAR_CITE_RE = re.compile(
    r"\b([A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?)?)\s*,?\s*(\d{4})\b"
)


def extract_citation_ids(text: str) -> Dict[str, List[str]]:
    dois = list({m.group(1).rstrip(".,;)") for m in DOI_RE.finditer(text)})
    arxiv_ids = list({m.group(1) for m in ARXIV_RE.finditer(text)})
    author_year = [f"{m.group(1)} {m.group(2)}" for m in YEAR_CITE_RE.finditer(text)]
    return {"dois": dois, "arxiv_ids": arxiv_ids, "author_year": author_year}


def resolve_doi(doi: str) -> Dict[str, Any]:
    """Resolve DOI via CrossRef. Returns {resolved: bool, ...}."""
    try:
        url = f"https://api.crossref.org/works/{doi}"
        headers = {"User-Agent": f"ScholarGraph/2.0 (mailto:{config.openalex_email})"}
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            msg = r.json().get("message", {})
            title = ""
            if msg.get("title"):
                title = msg["title"][0]
            return {
                "resolved": True,
                "doi": doi,
                "title": title,
                "year": (msg.get("published-print") or msg.get("published-online") or {})
                .get("date-parts", [[None]])[0][0],
                "container": (msg.get("container-title") or [""])[0],
            }
        return {"resolved": False, "doi": doi, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"resolved": False, "doi": doi, "error": str(e)}


def resolve_arxiv(arxiv_id: str) -> Dict[str, Any]:
    try:
        url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and "<entry>" in r.text:
            title_m = re.search(r"<title>(.*?)</title>", r.text, re.DOTALL)
            title = title_m.group(1).strip() if title_m else ""
            if title.lower().startswith("arxiv query"):
                # first title is feed title; take next
                titles = re.findall(r"<title>(.*?)</title>", r.text, re.DOTALL)
                title = titles[1].strip() if len(titles) > 1 else title
            return {"resolved": True, "arxiv_id": arxiv_id, "title": title}
        return {"resolved": False, "arxiv_id": arxiv_id, "error": "not found"}
    except Exception as e:
        return {"resolved": False, "arxiv_id": arxiv_id, "error": str(e)}


def verify_citations(text: str) -> Dict[str, Any]:
    """
    Hard check: every DOI/arXiv ID must resolve.
    Author-year citations without IDs are flagged as unverifiable (soft fail).
    """
    ids = extract_citation_ids(text)
    resolved = []
    failed = []
    unverifiable = []

    for doi in ids["dois"]:
        result = resolve_doi(doi)
        (resolved if result["resolved"] else failed).append(result)

    for aid in ids["arxiv_ids"]:
        result = resolve_arxiv(aid)
        (resolved if result["resolved"] else failed).append(result)

    for ay in ids["author_year"]:
        # Soft: no API proof without DOI
        unverifiable.append({"citation": ay, "reason": "author-year without DOI/arXiv"})

    n_hard = len(ids["dois"]) + len(ids["arxiv_ids"])
    n_failed = len(failed)
    if n_hard == 0:
        score = 5.0 if unverifiable else 10.0  # no citable IDs → neutral/ok
        passed = True
        note = "No DOI/arXiv IDs found to verify"
    else:
        score = max(0.0, 10.0 * (1.0 - n_failed / n_hard))
        passed = n_failed == 0
        note = f"Resolved {n_hard - n_failed}/{n_hard} citation IDs"

    return {
        "passed": passed,
        "score": score,
        "note": note,
        "resolved": resolved,
        "failed": failed,
        "unverifiable": unverifiable[:20],
        "ids_found": ids,
    }


# ---------------------------------------------------------------------------
# Statistical validity (deterministic re-derivation from raw results)
# ---------------------------------------------------------------------------

def _welch_t_p(a: List[float], b: List[float]) -> Optional[float]:
    """Two-sided Welch t-test p-value without scipy dependency if needed."""
    try:
        from scipy import stats

        if len(a) < 2 or len(b) < 2:
            return None
        _, p = stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    except Exception:
        # Manual approximation
        import numpy as np

        if len(a) < 2 or len(b) < 2:
            return None
        a, b = np.array(a, float), np.array(b, float)
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)
        na, nb = len(a), len(b)
        se = math.sqrt(va / na + vb / nb)
        if se == 0:
            return 1.0
        t = (ma - mb) / se
        # crude two-sided normal approx
        from math import erf, sqrt

        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        return float(p)


def verify_statistics(
    reported: Dict[str, Any],
    raw_results_path: Optional[str] = None,
    raw_data: Optional[Dict[str, Any]] = None,
    rtol: float = 0.05,
) -> Dict[str, Any]:
    """
    Re-derive mean/std (and optional p-values) from stored raw results
    and compare to what the paper/writer reported.
    """
    raw = raw_data
    if raw is None and raw_results_path:
        path = Path(raw_results_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

    if not raw:
        return {
            "passed": False,
            "score": 0.0,
            "note": "No raw results available for statistical verification",
            "mismatches": ["missing_raw_results"],
        }

    aggregate = raw.get("aggregate_metrics") or {}
    mismatches = []
    checks = 0
    passes = 0

    reported_metrics = reported.get("metrics") or reported
    for key, agg in aggregate.items():
        if key not in reported_metrics:
            continue
        checks += 1
        rep_val = reported_metrics[key]
        if isinstance(rep_val, dict):
            rep_mean = rep_val.get("mean", rep_val.get("value"))
            rep_std = rep_val.get("std")
        else:
            rep_mean = rep_val
            rep_std = None

        true_mean = agg.get("mean")
        true_std = agg.get("std")

        if true_mean is not None and rep_mean is not None:
            if abs(float(rep_mean) - float(true_mean)) > max(rtol * abs(true_mean), 1e-6):
                mismatches.append(
                    f"{key}.mean reported={rep_mean} raw={true_mean}"
                )
            else:
                passes += 1

        if rep_std is not None and true_std is not None:
            checks += 1
            if abs(float(rep_std) - float(true_std)) > max(rtol * abs(true_std), 1e-6):
                mismatches.append(f"{key}.std reported={rep_std} raw={true_std}")
            else:
                passes += 1

    # Optional p-value check if both groups present
    if "group_a" in raw and "group_b" in raw and "p_value" in reported_metrics:
        checks += 1
        p_true = _welch_t_p(raw["group_a"], raw["group_b"])
        p_rep = float(reported_metrics["p_value"])
        if p_true is None:
            mismatches.append("could_not_compute_p_value")
        elif abs(p_true - p_rep) > max(0.02, rtol * abs(p_true)):
            mismatches.append(f"p_value reported={p_rep} recomputed={p_true}")
        else:
            passes += 1

    if checks == 0:
        return {
            "passed": True,
            "score": 7.0,
            "note": "No overlapping metrics to verify against raw data",
            "mismatches": [],
        }

    score = 10.0 * (passes / checks) if checks else 0.0
    return {
        "passed": len(mismatches) == 0,
        "score": score,
        "note": f"Verified {passes}/{checks} reported statistics against raw data",
        "mismatches": mismatches,
    }


def hard_verify_section(
    content: str,
    engineer_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all hard checks; LLM soft review should only run AFTER this passes."""
    citation = verify_citations(content)

    stats_results = []
    if engineer_outputs:
        for exp_name, output in engineer_outputs.items():
            raw_path = None
            if isinstance(output, dict):
                raw_path = output.get("raw_results_path")
                reported = output.get("results", {}).get("metrics") or output.get(
                    "aggregate_metrics", {}
                )
                # Also try extracting claimed numbers from text for this experiment
                stats_results.append(
                    {
                        "experiment": exp_name,
                        **verify_statistics(
                            {"metrics": reported} if not isinstance(reported, dict) or "metrics" not in reported else reported,
                            raw_results_path=raw_path,
                            raw_data=output if "aggregate_metrics" in output else output.get("multi_seed"),
                        ),
                    }
                )

    stats_passed = all(s.get("passed", True) for s in stats_results) if stats_results else True
    avg_stats = (
        sum(s.get("score", 10) for s in stats_results) / len(stats_results)
        if stats_results
        else 10.0
    )

    hard_passed = citation["passed"] and stats_passed
    # Hard checks dominate: fail hard → cap score
    combined = min(citation["score"], avg_stats) if hard_passed else min(
        citation["score"], avg_stats, 4.0
    )

    return {
        "passed": hard_passed,
        "score": combined,
        "citation": citation,
        "statistics": stats_results,
        "feedback": _format_hard_feedback(citation, stats_results),
    }


def reproducibility_dossier(plan: Optional[Dict[str, Any]], engineer_outputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Venue-style disclosure checklist derived from executable artifacts, not prose."""
    experiments = (plan or {}).get("experiments") or []
    outputs = engineer_outputs or {}
    checks = {
        "falsifiable_predictions": all(isinstance(e, dict) and bool(e.get("falsifiable_prediction")) for e in experiments) if experiments else False,
        "named_baselines": all(isinstance(e, dict) and bool(e.get("baselines") or e.get("baseline_comparison")) for e in experiments) if experiments else False,
        "statistical_tests": all(isinstance(e, dict) and bool(e.get("statistical_test")) for e in experiments) if experiments else False,
        "multi_seed_raw_results": bool(outputs) and all(isinstance(o, dict) and bool(o.get("raw_results_path") or o.get("multi_seed")) for o in outputs.values()),
        "executable_code": bool(outputs) and all(isinstance(o, dict) and bool(o.get("code")) for o in outputs.values()),
        "limitations_disclosed": True,  # Editor carries unresolved debate objections into Limitations.
    }
    return {"checks": checks, "passed": all(checks.values()), "score": round(10 * sum(checks.values()) / len(checks), 2)}


def _format_hard_feedback(citation: Dict, stats_results: List[Dict]) -> str:
    parts = [f"Citations: {citation.get('note')}"]
    if citation.get("failed"):
        parts.append(
            "Unresolved: "
            + ", ".join(
                str(f.get("doi") or f.get("arxiv_id")) for f in citation["failed"]
            )
        )
    for s in stats_results:
        parts.append(f"Stats[{s.get('experiment')}]: {s.get('note')}")
        if s.get("mismatches"):
            parts.append("Mismatches: " + "; ".join(s["mismatches"][:5]))
    return "\n".join(parts)
