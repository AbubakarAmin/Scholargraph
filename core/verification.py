"""
Hard verification checks: citation resolution + statistical validity.
These are deterministic — not LLM vibe scores.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import config
from .llm import call_llm
from .utils import call_llm_json, parse_json_from_llm

logger = logging.getLogger(__name__)


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

CLAIM_TYPES = {"literature_reference", "method_definition", "planned_test", "empirical_result"}


def extract_citation_ids(text: str) -> Dict[str, List[str]]:
    dois = list({m.group(1).rstrip(".,;)") for m in DOI_RE.finditer(text)})
    arxiv_ids = list({m.group(1) for m in ARXIV_RE.finditer(text)})
    author_year = [f"{m.group(1)} {m.group(2)}" for m in YEAR_CITE_RE.finditer(text)]
    return {"dois": dois, "arxiv_ids": arxiv_ids, "author_year": author_year}


def extract_citation_metadata(text: str) -> Dict[str, Dict[str, Any]]:
    """Extract nearby title/author fields from prose and BibTeX-like entries."""
    metadata: Dict[str, Dict[str, Any]] = {}
    blocks = re.split(r"(?=@\w+\{)|\n\s*\n", text)
    for block in blocks:
        identifiers = extract_citation_ids(block)
        title_match = re.search(r"(?:title\s*[=:]\s*\{?|\"|')([^}\"'\n]+)", block, re.I)
        author_match = re.search(r"author\s*[=:]\s*\{?([^}\n]+)", block, re.I)
        title = title_match.group(1).strip() if title_match else ""
        authors = [part.strip() for part in re.split(r"\s+and\s+|,", author_match.group(1)) if part.strip()] if author_match else []
        for identifier in identifiers["dois"] + identifiers["arxiv_ids"]:
            if title or authors:
                metadata[identifier] = {"title": title, "authors": authors}
    return metadata


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
                "authors": [author.get("family") or author.get("literal") or "" for author in (msg.get("author") or [])],
                "year": (msg.get("published-print") or msg.get("published-online") or {})
                .get("date-parts", [[None]])[0][0],
                "container": (msg.get("container-title") or [""])[0],
            }
        return {"resolved": False, "doi": doi, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"resolved": False, "doi": doi, "error": str(e)}


def resolve_arxiv(arxiv_id: str) -> Dict[str, Any]:
    for attempt in range(3):
        try:
            url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
            r = requests.get(url, timeout=15)
            if r.status_code == 429 and attempt < 2:
                retry_after = float(r.headers.get("Retry-After", "5"))
                time.sleep(min(30.0, max(retry_after, 5.0 * (2 ** attempt))))
                continue
            if r.status_code == 200 and "<entry>" in r.text:
                title_m = re.search(r"<title>(.*?)</title>", r.text, re.DOTALL)
                title = title_m.group(1).strip() if title_m else ""
                if title.lower().startswith("arxiv query"):
                    titles = re.findall(r"<title>(.*?)</title>", r.text, re.DOTALL)
                    title = titles[1].strip() if len(titles) > 1 else title
                authors = re.findall(r"<name>(.*?)</name>", r.text, re.DOTALL)
                return {"resolved": True, "arxiv_id": arxiv_id, "title": title, "authors": [a.strip() for a in authors]}
            return {"resolved": False, "arxiv_id": arxiv_id, "error": f"HTTP {r.status_code}"}
        except Exception as e:
            if attempt < 2:
                time.sleep(2.0 * (2 ** attempt))
                continue
            return {"resolved": False, "arxiv_id": arxiv_id, "error": str(e)}
    return {"resolved": False, "arxiv_id": arxiv_id, "error": "max retries exceeded"}


def _normalize_words(value: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", (value or "").lower()) if len(word) > 2}


def _metadata_diff(result: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    """Compare writer-supplied bibliography metadata with resolved metadata."""
    expected_title = expected.get("title") or ""
    expected_authors = expected.get("authors") or []
    differences: Dict[str, Any] = {}
    if expected_title:
        actual = _normalize_words(result.get("title", ""))
        wanted = _normalize_words(expected_title)
        if not wanted or actual != wanted:
            differences["title"] = {
                "expected": expected_title,
                "actual": result.get("title", ""),
            }
    if expected_authors:
        actual = _normalize_words(" ".join(result.get("authors") or []))
        wanted = _normalize_words(" ".join(expected_authors) if isinstance(expected_authors, list) else expected_authors)
        if not wanted or not wanted.issubset(actual):
            differences["authors"] = {
                "expected": expected_authors,
                "actual": result.get("authors") or [],
            }
    return differences


def _metadata_matches(result: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Compatibility predicate for callers that only need a boolean."""
    return not _metadata_diff(result, expected)


def verify_citations(text: str, citation_metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Hard check: every DOI/arXiv ID must resolve.
    Author-year citations without IDs are flagged as unverifiable (soft fail).
    """
    ids = extract_citation_ids(text)
    citation_metadata = citation_metadata if citation_metadata is not None else extract_citation_metadata(text)
    resolved = []
    failed = []
    mismatched = []
    unverifiable = []

    for doi in ids["dois"]:
        result = resolve_doi(doi)
        expected = citation_metadata.get(doi) if citation_metadata else None
        if result["resolved"] and expected:
            differences = _metadata_diff(result, expected)
            if differences:
                result = {
                    **result,
                    "error": "resolved metadata does not match claimed bibliography",
                    "metadata_mismatch": True,
                    "metadata_diff": differences,
                    "expected_metadata": expected,
                }
                mismatched.append(result)
        (resolved if result["resolved"] and not result.get("metadata_mismatch") else failed).append(result)

    for aid in ids["arxiv_ids"]:
        result = resolve_arxiv(aid)
        expected = citation_metadata.get(aid) if citation_metadata else None
        if result["resolved"] and expected:
            differences = _metadata_diff(result, expected)
            if differences:
                result = {
                    **result,
                    "error": "resolved metadata does not match claimed bibliography",
                    "metadata_mismatch": True,
                    "metadata_diff": differences,
                    "expected_metadata": expected,
                }
                mismatched.append(result)
        (resolved if result["resolved"] and not result.get("metadata_mismatch") else failed).append(result)

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
        "metadata_mismatches": mismatched,
        "unverifiable": unverifiable[:20],
        "ids_found": ids,
    }


def classify_claim(text: str, evidence_artifact_ids: Optional[List[str]] = None) -> str:
    """Classify a claim conservatively before it enters the evidence ledger."""
    lowered = text.lower()
    if re.search(r"\b(doi|arxiv|et al\.?|\(\d{4}\))", lowered):
        return "literature_reference"
    if any(token in lowered for token in ("we define", "we use", "algorithm", "methodology", "equation")):
        return "method_definition"
    if any(token in lowered for token in ("will evaluate", "plan to", "we propose to", "future experiment")):
        return "planned_test"
    if evidence_artifact_ids and re.search(r"\d+(?:\.\d+)?\s*%|\b(mean|std|accuracy|f1|mse|p\s*[<=>])\b", lowered):
        return "empirical_result"
    return "method_definition"


def cross_section_numeric_consistency(sections: Dict[str, str], engineer_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Reject artifact or cross-section conflicts for one experiment metric."""
    outputs = engineer_outputs or {}
    expected: Dict[str, Dict[str, float]] = {}
    for experiment, output in outputs.items():
        metrics = (output or {}).get("aggregate_metrics") or (output or {}).get("results", {}).get("metrics") or {}
        expected[experiment] = {key: float(value.get("mean", value) if isinstance(value, dict) else value) for key, value in metrics.items() if isinstance(value, (int, float, dict))}
    claims: Dict[tuple[str, str, float], List[str]] = {}
    for section, content in sections.items():
        for experiment, metrics in expected.items():
            for metric in metrics:
                pattern = rf"\b{re.escape(metric)}\b[^\d%]{{0,40}}(\d+(?:\.\d+)?)\s*(%)?"
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    value = float(match.group(1)) / 100 if match.group(2) else float(match.group(1))
                    claims.setdefault((experiment, metric, value), []).append(section)
    conflicts = []
    for experiment, metrics in expected.items():
        for metric in metrics:
            artifact_value = metrics[metric]
            observed = [(value, sections_seen) for (exp, name, value), sections_seen in claims.items() if exp == experiment and name == metric]
            claim_values = {value for value, _ in observed}
            if observed and any(not math.isclose(value, artifact_value, rel_tol=0.005, abs_tol=1e-9) for value in claim_values):
                observed.append((artifact_value, ["structured_artifact"]))
                conflicts.append({
                    "experiment": experiment,
                    "metric": metric,
                    "claims": [{"value": value, "sections": sections_seen} for value, sections_seen in observed],
                    "reason": "manuscript claim disagrees with structured artifact",
                })
            elif len({value for value, _ in observed}) > 1:
                conflicts.append({
                    "experiment": experiment,
                    "metric": metric,
                    "claims": [{"value": value, "sections": sections_seen} for value, sections_seen in observed],
                    "reason": "sections disagree",
                })
    collected = [
        {"experiment": experiment, "metric": metric, "value": value, "sections": sections_seen}
        for (experiment, metric, value), sections_seen in claims.items()
    ]
    return {
        "passed": not conflicts,
        "claims": collected,
        "conflicts": conflicts,
        "note": "No numeric conflicts" if not conflicts else "Conflicting numeric claims",
    }


def preregister_power(planned_effect_size: float, alpha: float = 0.05, target_power: float = 0.8) -> Dict[str, Any]:
    """Compute a prospective two-group sample-size requirement before execution."""
    from scipy import stats

    if planned_effect_size <= 0 or not 0 < alpha < 1 or not 0 < target_power < 1:
        raise ValueError("effect size must be positive; alpha and target_power must be in (0, 1)")
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_power = float(stats.norm.ppf(target_power))
    required_n = int(math.ceil(2 * ((z_alpha + z_power) / planned_effect_size) ** 2))
    return {"planned_effect_size": float(planned_effect_size), "alpha": float(alpha), "target_power": float(target_power), "required_n_per_group": required_n, "method": "two-group normal approximation"}


def validate_reviewer_checklist(sections: Dict[str, str], engineer_outputs: Optional[Dict[str, Any]] = None, plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Enforce publication checklist requirements deterministically."""
    text = "\n".join(sections.values())
    outputs = engineer_outputs or {}
    checks = {
        "limitations": bool(sections.get("Limitations", "").strip()),
        "baselines_declared": all(bool(exp.get("baselines") or exp.get("baseline_comparison")) for exp in (plan or {}).get("experiments", []) if isinstance(exp, dict)),
        "outcomes_reported": bool(outputs) and all(bool(item.get("outcome") or item.get("aggregate_metrics") or item.get("results")) for item in outputs.values() if isinstance(item, dict)),
        "uncertainty_or_sample_size": not bool(re.search(r"\d+(?:\.\d+)?\s*%|\b(?:accuracy|f1|mse)\b", text, re.I)) or bool(re.search(r"\b(?:n\s*=|std|confidence interval|ci\b)", text, re.I)),
        "literature_evidence": not bool(re.search(r"\b(?:doi|arxiv|et al\.?|\(\d{4}\))", text, re.I)) or bool(re.search(r"retrieved|abstract|source text|literature evidence", text, re.I)),
    }
    return {"passed": all(checks.values()), "checks": checks, "failed": [name for name, passed in checks.items() if not passed]}


def novelty_overlap_check(
    sections: Dict[str, str],
    topic: Optional[Dict[str, Any]] = None,
    threshold: float = 0.65,
) -> Dict[str, Any]:
    """Deterministic novelty-plagiarism screen (Gupta & Pruthi, ACL 2025).

    ~24% of AI-generated research documents borrow heavily from prior work
    without acknowledgment. This compares the manuscript's framing text
    (Abstract + Introduction) against the closest prior work and retrieved
    literature evidence using a content-word overlap coefficient.
    Overlap coefficient = |shared| / min(|draft|, |prior|), which is more
    sensitive than Jaccard to a small draft closely paraphrasing one source.
    """
    prior_sources: List[Dict[str, str]] = []
    structured = (topic or {}).get("structured_hypothesis") or {}
    closest = structured.get("closest_prior_work") if isinstance(structured, dict) else None
    if isinstance(closest, dict):
        text = " ".join(str(closest.get(key) or "") for key in ("title", "abstract", "summary", "contribution") if closest.get(key))
        if text.strip():
            prior_sources.append({"source": "closest_prior_work", "text": text})
    elif isinstance(closest, str) and closest.strip():
        prior_sources.append({"source": "closest_prior_work", "text": closest})
    for paper in (topic or {}).get("literature_evidence") or []:
        if not isinstance(paper, dict):
            continue
        text = " ".join(
            str(paper.get(key) or "") for key in ("title", "abstract")
        ).strip()
        if text:
            prior_sources.append({"source": paper.get("title") or paper.get("doi") or paper.get("arxiv_id") or "literature", "text": text})

    framing = "\n".join(
        sections.get(name, "") for name in ("Abstract", "Introduction")
        if isinstance(sections.get(name), str)
    )
    if not prior_sources or not framing.strip():
        return {
            "passed": True,
            "max_overlap": 0.0,
            "threshold": threshold,
            "overlaps": [],
            "findings": [],
            "note": "no prior sources to compare against",
        }

    def content_tokens(text: str) -> set:
        return {word for word in re.findall(r"[a-z]{4,}", (text or "").lower())}

    draft_tokens = content_tokens(framing)
    overlaps: List[Dict[str, Any]] = []
    for source in prior_sources:
        source_tokens = content_tokens(source["text"])
        if not source_tokens:
            continue
        shared = draft_tokens & source_tokens
        denominator = min(len(draft_tokens), len(source_tokens))
        coefficient = (len(shared) / denominator) if denominator else 0.0
        overlaps.append({"source": source["source"], "overlap_coefficient": round(coefficient, 3), "shared_terms": len(shared)})
    max_overlap = max((item["overlap_coefficient"] for item in overlaps), default=0.0)
    flagged = [item for item in overlaps if item["overlap_coefficient"] > threshold]
    return {
        "passed": not flagged,
        "max_overlap": max_overlap,
        "threshold": threshold,
        "overlaps": sorted(overlaps, key=lambda item: item["overlap_coefficient"], reverse=True)[:5],
        "note": (
            f"Max framing overlap {max_overlap:.2f} vs threshold {threshold}"
            if overlaps else "no prior sources to compare against"
        ),
        "findings": [
            {"source": item["source"], "overlap_coefficient": item["overlap_coefficient"]}
            for item in overlaps if item["overlap_coefficient"] > threshold
        ],
    }


def consistency_referee(
    sections: Dict[str, str],
    plan: Optional[Dict[str, Any]] = None,
    engineer_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Use one isolated model pass to find contradictions in the full draft."""
    """Use one isolated model pass to find contradictions in the full draft."""
    prompt = f"""
Read this assembled research manuscript as a consistency referee.
Your only task is to identify internal contradictions in numbers, datasets,
methods, outcomes, or contribution framing. Do not rewrite prose and do not
infer missing evidence. Return JSON only:
{{"findings": [{{"category": "numeric|dataset|method|outcome|contribution", "message": "...", "blocking": true}}]}}

Plan and structured evidence:
{json.dumps({"plan": plan or {}, "engineer_outputs": engineer_outputs or {}}, sort_keys=True, default=str)[:12000]}

Assembled sections:
{json.dumps(sections, sort_keys=True, default=str)[:30000]}
"""
    try:
        parsed = call_llm_json(
            prompt,
            temperature=0.0,
            tier="judge",
            max_tokens=2500,
            attempts=2,
            call_fn=call_llm,
        )
    except Exception as exc:
        return {"passed": False, "findings": [{"category": "referee_error", "message": "Consistency referee failed to execute", "blocking": True, "error_type": type(exc).__name__}]}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("findings"), list):
        return {"passed": False, "findings": [{"category": "referee_error", "message": "Consistency referee returned malformed output", "blocking": True}]}
    findings = [item for item in parsed["findings"] if isinstance(item, dict)]
    blocking = [item for item in findings if item.get("blocking", True)]
    return {"passed": not blocking, "findings": findings}


def final_manuscript_referee(
    sections: Dict[str, str],
    plan: Optional[Dict[str, Any]] = None,
    engineer_outputs: Optional[Dict[str, Any]] = None,
    topic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run release-blocking checks over the assembled manuscript as one document."""
    text = "\n".join(sections.values())
    numeric = cross_section_numeric_consistency(sections, engineer_outputs)
    citation = verify_citations(text)
    checklist = validate_reviewer_checklist(sections, engineer_outputs, plan)
    model_referee = consistency_referee(sections, plan, engineer_outputs)
    novelty = novelty_overlap_check(sections, topic)
    findings = []
    if not citation["passed"]:
        findings.append({"check": "citations", "details": citation.get("failed", [])})
    if not numeric["passed"]:
        findings.append({"check": "numeric_consistency", "details": numeric["conflicts"]})
    if not checklist["passed"]:
        findings.append({"check": "reviewer_checklist", "details": checklist["failed"]})
    if not model_referee["passed"]:
        findings.append({"check": "consistency_referee", "details": model_referee["findings"]})
    if not novelty["passed"]:
        findings.append({"check": "novelty_overlap", "details": novelty["findings"]})
    prohibited = [phrase for phrase in PROHIBITED_MANUSCRIPT_TEXT if phrase in text.lower()]
    if prohibited:
        findings.append({"check": "harness_diagnostics", "details": prohibited})
    if not any(name.lower() == "limitations" and content.strip() for name, content in sections.items()):
        findings.append({"check": "limitations", "details": "A non-empty Limitations section is required"})
    quantitative = re.search(r"\d+(?:\.\d+)?\s*%|\b(?:accuracy|f1|mse|p\s*[<=>])\b", text, re.I)
    if quantitative and not re.search(r"\b(?:n\s*=|std|confidence interval|ci\b)", text, re.I):
        findings.append({"check": "uncertainty_reporting", "details": "Quantitative claims require n, standard deviation, or confidence interval"})
    declared_datasets = {
        str((experiment.get("dataset") or {}).get("name"))
        for experiment in (plan or {}).get("experiments", [])
        if isinstance(experiment, dict) and (experiment.get("dataset") or {}).get("name")
    }
    if declared_datasets:
        missing = [dataset for dataset in declared_datasets if dataset.lower() not in text.lower()]
        if missing:
            findings.append({"check": "dataset_consistency", "details": missing})
    return {
        "passed": not findings,
        "findings": findings,
        "citation": citation,
        "numeric": numeric,
        "checklist": checklist,
        "consistency_referee": model_referee,
        "novelty": novelty,
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
    *,
    section_name: Optional[str] = None,
    content_requirements: Optional[str] = None,
    min_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """Run all hard checks; LLM soft review should only run AFTER this passes."""
    from .utils import is_degenerate_llm_output, strip_markdown_headers

    section_minima = {
        "abstract": 120,
        "introduction": 400,
        "related work": 400,
        "methods": 300,
        "experiments": 250,
        "results": 200,
        "discussion": 250,
        "conclusion": 150,
        "limitations": 120,
    }
    required_chars = min_chars
    if required_chars is None:
        required_chars = section_minima.get((section_name or "").lower(), 200)

    body = strip_markdown_headers(content or "")
    substance_errors: List[str] = []
    if is_degenerate_llm_output(content or "", min_chars=required_chars):
        substance_errors.append(
            f"section content is empty, a safety stub, or below minimum length ({required_chars} chars)"
        )
    elif len(body) < required_chars:
        substance_errors.append(
            f"section body length {len(body)} is below minimum {required_chars}"
        )
    if content_requirements:
        req_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z]{4,}", str(content_requirements))
            if token.lower() not in {"this", "that", "with", "from", "section", "should", "include"}
        ][:8]
        lowered = body.lower()
        missing = [token for token in req_tokens if token not in lowered]
        # Only fail when almost none of the requirement tokens appear — avoid brittle exact matching.
        if req_tokens and len(missing) >= max(3, int(0.75 * len(req_tokens))):
            substance_errors.append(
                "section does not address content_requirements (missing key terms: "
                + ", ".join(missing[:5])
                + ")"
            )

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

    substance_passed = not substance_errors
    hard_passed = citation["passed"] and stats_passed and substance_passed
    # Hard checks dominate: fail hard → cap score
    combined = min(citation["score"], avg_stats) if hard_passed else min(
        citation["score"], avg_stats, 4.0
    )
    if not substance_passed:
        combined = min(combined, 3.0)

    feedback = _format_hard_feedback(citation, stats_results)
    if substance_errors:
        feedback = "SUBSTANCE: " + "; ".join(substance_errors) + "\n" + feedback

    return {
        "passed": hard_passed,
        "score": combined,
        "citation": citation,
        "statistics": stats_results,
        "substance_errors": substance_errors,
        "feedback": feedback,
    }


def reproducibility_dossier(plan: Optional[Dict[str, Any]], engineer_outputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Venue-style disclosure checklist derived from executable artifacts, not prose."""
    experiments = (plan or {}).get("experiments") or []
    outputs = engineer_outputs or {}
    has_plan = bool(plan) and bool(experiments)
    has_outputs = bool(outputs)
    checks = {
        "falsifiable_predictions": all(isinstance(e, dict) and bool(e.get("falsifiable_prediction")) for e in experiments) if experiments else False,
        "named_baselines": all(isinstance(e, dict) and bool(e.get("baselines") or e.get("baseline_comparison")) for e in experiments) if experiments else False,
        "statistical_tests": all(isinstance(e, dict) and bool(e.get("statistical_test")) for e in experiments) if experiments else False,
        "multi_seed_raw_results": bool(outputs) and all(isinstance(o, dict) and bool(o.get("raw_results_path") or o.get("multi_seed")) and (not o.get("raw_results_path") or Path(str(o["raw_results_path"])).is_file()) for o in outputs.values()),
        "executable_code": bool(outputs) and all(isinstance(o, dict) and bool(o.get("code")) and o.get("success", True) for o in outputs.values()),
        # Never pass by default when there is nothing to evaluate.
        "contract_provenance": has_outputs and all(
            isinstance(o, dict) and bool(o.get("contract_hash")) for o in outputs.values()
        ),
        "limitations_disclosed": has_plan and has_outputs,
    }
    return {"checks": checks, "passed": all(checks.values()), "score": round(10 * sum(checks.values()) / len(checks), 2)}


PROHIBITED_MANUSCRIPT_TEXT = (
    "tracemalloc",
    "sandbox blocked",
    "import blocked",
    "syntaxerror",
    "traceback",
    "object has no attribute",
    "experiment_failed",
    "plan_revision_requested",
)


def validate_empirical_claims(content: str, engineer_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Reject leaked harness diagnostics from empirical manuscript sections."""
    lowered = content.lower()
    prohibited = [phrase for phrase in PROHIBITED_MANUSCRIPT_TEXT if phrase in lowered]
    successful = [
        name for name, output in (engineer_outputs or {}).items()
        if isinstance(output, dict) and output.get("success")
        or isinstance(output, dict) and output.get("aggregate_metrics")
    ]
    return {
        "passed": not prohibited and bool(successful),
        "prohibited_text": prohibited,
        "successful_experiments": successful,
        "note": "Empirical text is grounded in completed experiment outputs" if successful and not prohibited else "Empirical text contains unsupported harness text or has no completed experiment",
    }


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
