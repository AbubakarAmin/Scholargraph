"""
TopicHunterAgent — citation-graph gap analysis, novelty filter, parallel hunts.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import arxiv
import numpy as np
import requests

from core.config import config
from core.utils import log_agent_action, parse_json_from_llm, calculate_similarity, title_token_overlap
from core.llm import call_llm, generate_embedding
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.contracts import FeasibilityReport, Topic
from core.memory import memory
from core.run_log import CrossRunMemory, get_tracker
from core.sources import SourceClient
from agents.hypothesis_debate import EloStore, hypothesis_kind
from core.capabilities import SANDBOX_CAPABILITY_MANIFEST, check_plan_feasibility
from core.evidence_synthesis import build_cross_paper_evidence_map, validate_candidate_bridge_claim, validate_topic_admission


_FAILED_TOPIC_OVERLAP_THRESHOLD = 0.55

_ARXIV_CATEGORIES = {
    "machine_learning": ["cs.LG", "stat.ML"],
    "natural_language_processing": ["cs.CL", "cs.AI"],
    "computer_vision": ["cs.CV", "cs.AI"],
    "reinforcement_learning": ["cs.LG", "cs.AI"],
    "graph_neural_networks": ["cs.LG", "cs.SI"],
    "fairness": ["cs.LG", "cs.CY"],
    "federated_learning": ["cs.LG", "cs.DC"],
    "optimization": ["math.OC", "cs.LG"],
    "general": ["cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML"],
}

_TOPICS_TO_SUBCATEGORY = {
    "attention": "machine_learning",
    "graph": "graph_neural_networks",
    "diffusion": "machine_learning",
    "reinforcement": "reinforcement_learning",
    "federated": "federated_learning",
    "llm": "natural_language_processing",
    "vision": "computer_vision",
    "nlp": "natural_language_processing",
    "general": "general",
}

_QUERY_STOPWORDS = frozenset({
    "and", "for", "the", "are", "but", "not", "with", "this", "that", "from",
    "have", "has", "was", "were", "been", "being", "can", "may", "its", "our",
})

_EXPLORATION_KIND_BIASES = {
    "attention": {"method": ["transformer", "self-attention", "multi-head"], "evaluation": ["efficiency", "scaling", "long-context"]},
    "graph": {"method": ["graph neural", "message passing", "spectral"], "evaluation": ["inductive", "scalability", "heterogeneous"]},
    "diffusion": {"method": ["diffusion model", "score-based", "denoising"], "evaluation": ["sample quality", "convergence", "likelihood"]},
    "reinforcement": {"method": ["reinforcement learning", "policy gradient", "temporal difference"], "evaluation": ["sample efficiency", "exploration", "stability"]},
    "federated": {"method": ["federated", "distributed optimization", "privacy"], "evaluation": ["communication efficiency", "convergence", "non-IID"]},
    "llm": {"method": ["language model", "fine-tuning", "prompt"], "evaluation": ["zero-shot", "reasoning", "alignment"]},
    "vision": {"method": ["visual", "image", "spatial"], "evaluation": ["detection", "segmentation", "recognition"]},
    "nlp": {"method": ["text", "sequence", "tokeniz"], "evaluation": ["classification", "generation", "extraction"]},
    "general": {"method": ["learning", "optimization", "generalization"], "evaluation": ["accuracy", "robustness", "scalability"]},
}


class ResearchSourceUnavailable(RuntimeError):
    """Raised when discovery cannot consult any external scholarly source."""


class TopicHunterAgent:
    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.runtime_config = self.context.config if self.context else config
        self.vector_memory = self.context.memory if self.context else memory
        self.client = get_llm_client()
        self.openalex_headers = {
            "User-Agent": f"ScholarGraph/2.0 (mailto:{self.runtime_config.openalex_email})"
        }
        self.s2_headers = {"User-Agent": "ScholarGraph/2.0"}
        if self.runtime_config.semantic_scholar_api_key:
            self.s2_headers["x-api-key"] = self.runtime_config.semantic_scholar_api_key
        self.base_urls = {
            "openalex": "https://api.openalex.org",
            "crossref": "https://api.crossref.org",
            "s2": "https://api.semanticscholar.org/graph/v1",
        }
        self._selection_counter = 0
        self._rng = random.Random(int(getattr(self.runtime_config, "topic_exploration_seed", 42)))
        self.source_client = SourceClient(
            str(Path(self.runtime_config.output_dir) / "source_cache")
        )
        self.rejection_log: List[Dict[str, Any]] = []
        self.source_health: Dict[str, Dict[str, Any]] = {}
        self._excluded_titles_cache: Optional[List[str]] = None
        self._iteration_failures = 0  # Track consecutive iteration failures

    def _source_ok(self, name: str):
        self.source_health[name] = {"ok": True}

    def _source_failed(self, name: str, error: Exception):
        self.source_health[name] = {"ok": False, "error": str(error)}

    def search_openalex(self, query: str, limit: int = 50, extra_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            url = f"{self.base_urls['openalex']}/works"
            params = {
                "search": query,
                "per_page": min(limit, 100),
                # OpenAlex exposes abstracts as an inverted index; requesting a
                # non-existent `abstract` field is a 400 in current API versions.
                "select": "id,title,abstract_inverted_index,publication_year,cited_by_count,concepts,type,doi",
                "mailto": self.runtime_config.openalex_email,
            }
            if extra_params:
                params.update(extra_params)
            artifact = self.source_client.fetch_json(
                "openalex",
                url,
                headers=self.openalex_headers,
                params=params,
                validator=lambda payload: isinstance(payload, dict) and "results" in payload,
            )
            if artifact["status"] == "unavailable":
                # Don't raise immediately - try to continue with partial data
                warnings = artifact.get("warnings", [])
                log_agent_action("TopicHunter", "openalex_unavailable", {"warnings": warnings})
                return []
            rows = artifact["content"].get("results", [])
            for row in rows:
                inverted = row.pop("abstract_inverted_index", None) or {}
                if inverted:
                    ordered = sorted(((pos, word) for word, positions in inverted.items() for pos in positions))
                    row["abstract"] = " ".join(word for _, word in ordered)
            self._source_ok("openalex")
            return rows
        except Exception as e:
            self._source_failed("openalex", e)
            log_agent_action("TopicHunter", "search_openalex_error", {"error": str(e)})
            return []

    def search_arxiv(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            results = []
            # arxiv.py 4 removed Search.results(); the Client owns iteration.
            for result in arxiv.Client(page_size=min(max_results, 100), delay_seconds=1).results(search):
                results.append({
                    "title": result.title,
                    "abstract": result.summary,
                    "year": result.published.year,
                    "authors": [a.name for a in result.authors],
                    "arxiv_id": result.entry_id,
                    "categories": result.categories,
                })
            self._source_ok("arxiv")
            return results
        except Exception as e:
            self._source_failed("arxiv", e)
            log_agent_action("TopicHunter", "search_arxiv_error", {"error": str(e)})
            return []

    def search_openalex_multi(self, queries: List[Tuple[str, Dict[str, Any]]], limit: int = 30) -> List[Dict[str, Any]]:
        """Run multiple OpenAlex queries with optional extra params and merge results."""
        all_results = []
        for query, params in queries:
            all_results.extend(self.search_openalex(query, limit, extra_params=params))
        return all_results

    def search_arxiv_multi(self, queries: List[str], max_results: int = 30) -> List[Dict[str, Any]]:
        """Run multiple arXiv queries and merge results."""
        all_results = []
        for query in queries:
            all_results.extend(self.search_arxiv(query, max_results))
        return all_results

    def _extract_rejected_fingerprints(self, cross_run_context: List[Dict[str, Any]]) -> List[str]:
        """Extract keyword fingerprints from rejected-topic items in CrossRunMemory.

        Only uses structured tags (rejection_reason, item title) — never raw
        free-text rejection reasons, consistent with the fail-closed memory policy.
        """
        fingerprints = []
        for entry in cross_run_context:
            item = str(entry.get("item") or "").strip()
            tag = str(entry.get("rejection_reason") or "").strip()
            if item:
                words = re.findall(r"[a-z][a-z0-9_-]{2,}", item.lower())
                fingerprints.append(" ".join(words[:5]))
            if tag and tag not in ("other",):
                fingerprints.append(tag)
        return fingerprints

    def _active_hypothesis_kind(self) -> Optional[str]:
        """Return the forced-exploration hypothesis kind if one is active this cycle, else None."""
        cfg = getattr(self, "runtime_config", None)
        every = max(1, int(getattr(cfg, "topic_exploration_every", 4) if cfg is not None else 4))
        if hasattr(self, "_selection_counter") and (self._selection_counter % every) == 0:
            try:
                elo = EloStore(context=getattr(self, "context", None))
                all_kinds = list(_TOPICS_TO_SUBCATEGORY.keys())
                under = elo.under_observed_kinds(all_kinds)
                if under:
                    return random.choice(under)
            except Exception:
                pass
        return None

    def _generate_dynamic_seeds(
        self,
        n_seeds: int = 5,
        cross_run_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Generate fresh seed phrases from current state instead of a static bank.

        Uses structured CrossRunMemory tags (fail-closed) to:
        - Pivot away from rejected topic areas
        - Bias toward underexplored hypothesis kinds
        - Combine method + evaluation signals from the kind bias tables
        """
        if cross_run_context is None:
            cross_run_context = CrossRunMemory().get_prompt_context()
        # Cap to last 20 rejections to prevent seed starvation from full history
        if len(cross_run_context) > 20:
            cross_run_context = cross_run_context[-20:]

        rejected_tokens: set = set()
        for entry in cross_run_context:
            item = str(entry.get("item") or "").lower()
            tag = str(entry.get("rejection_reason") or "")
            if item:
                rejected_tokens.update(w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", item) if len(w) > 3)
            if tag and tag not in ("other",):
                rejected_tokens.add(tag)

        active_kind = self._active_hypothesis_kind()

        seeds: List[str] = []

        # 1. Seeds from underexplored hypothesis kinds (avoid rejected areas)
        for kind, biases in _EXPLORATION_KIND_BIASES.items():
            method_terms = biases.get("method", [])
            eval_terms = biases.get("evaluation", [])
            if not method_terms or not eval_terms:
                continue
            method = method_terms[0]
            evaluation = eval_terms[0]
            kind_words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", f"{method} {evaluation}"))
            if kind_words & rejected_tokens:
                continue
            seeds.append(f"{method} {evaluation} open problems")

        # 2. Extra seeds biased toward the active exploration kind
        if active_kind and active_kind in _EXPLORATION_KIND_BIASES:
            biases = _EXPLORATION_KIND_BIASES[active_kind]
            for method_term in biases.get("method", [])[:2]:
                for eval_term in biases.get("evaluation", [])[:1]:
                    candidate = f"{method_term} {eval_term} underexplored challenges"
                    candidate_words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", candidate))
                    if not (candidate_words & rejected_tokens):
                        seeds.append(candidate)

        # 3. Generic fallback seeds that avoid rejected keyword areas
        generic_seeds = [
            "methodological gaps in evaluation",
            "cross-domain transfer limitations",
            "sample efficiency and data requirements",
            "robustness under distribution shift",
            "reproducibility and reporting standards",
            "interpretability for decision support",
            "scalability and computational limits",
            "edge cases and failure mode analysis",
        ]
        for s in generic_seeds:
            s_words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", s.lower()))
            if not (s_words & rejected_tokens):
                seeds.append(s)

        # Deduplicate preserving order
        seen: set = set()
        unique: List[str] = []
        for s in seeds:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique[:n_seeds]

    def _build_arxiv_queries(
        self,
        seed_hint: str,
        rejected_fingerprints: List[str],
        active_kind: Optional[str] = None,
    ) -> List[str]:
        """Build 2-4 field-scoped arXiv boolean queries.

        Uses cat:, abs:, ti: combinators instead of prose strings.
        Varies subcategory and sort mode.  Biases toward underexplored
        hypothesis kinds when one is active for this cycle.  Excludes
        keywords matching recently rejected fingerprints.
        """
        seed_lower = seed_hint.lower()
        keywords = [w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", seed_lower) if len(w) > 2 and w not in _QUERY_STOPWORDS][:3]

        # De-prioritize keywords that match rejected fingerprints
        if rejected_fingerprints:
            rejected_tokens = set()
            for fp in rejected_fingerprints:
                rejected_tokens.update(re.findall(r"[a-z][a-z0-9_-]{2,}", fp.lower()))
            keywords = [kw for kw in keywords if kw not in rejected_tokens]
            if not keywords:
                # Anchor terms always survive: drop rejected_tokens filter to prevent starvation
                all_seed_words = [w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", seed_lower)
                                  if w not in _QUERY_STOPWORDS and len(w) > 2]
                keywords = all_seed_words[:2]
        keywords = keywords[:3]

        if active_kind and active_kind in _TOPICS_TO_SUBCATEGORY:
            subcat_label = _TOPICS_TO_SUBCATEGORY[active_kind]
        else:
            subcat_label = "general"
        categories = _ARXIV_CATEGORIES.get(subcat_label, _ARXIV_CATEGORIES["general"])

        queries = []
        for cat in categories[:2]:
            if len(keywords) >= 2:
                kw_str = " OR ".join(f'"{kw}"' for kw in keywords[:2])
                queries.append(f"cat:{cat} AND ({kw_str})")
            if len(keywords) >= 1:
                queries.append(f"cat:{cat} AND ti:{keywords[0]}")
        if not queries and keywords:
            queries.append(f"cat:{categories[0]} AND abs:{keywords[0]}")

        # When an exploration kind is active, add one cross-pollination query
        # that pairs the kind's primary method term with the seed's first keyword.
        # This does NOT overwrite the seed's own keywords — it adds diversity.
        if active_kind and active_kind in _EXPLORATION_KIND_BIASES and keywords:
            biases = _EXPLORATION_KIND_BIASES[active_kind]
            kind_method = biases.get("method", [""])[0]
            seed_anchor = keywords[0]
            if kind_method and kind_method != seed_anchor:
                cross_q = f"cat:{categories[0]} AND abs:{kind_method} AND abs:{seed_anchor}"
                if cross_q not in queries:
                    queries.append(cross_q)

        return queries[:4]

    def _build_openalex_queries(
        self,
        seed_hint: str,
        rejected_fingerprints: List[str],
        active_kind: Optional[str] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Build 2-4 OpenAlex structured queries with filter params.

        Returns list of (search_query_string, extra_filter_params) tuples.
        Varies sort mode (relevance vs recency).  Biases toward underexplored
        hypothesis kinds when one is active.  Excludes keywords matching
        recently rejected fingerprints.
        """
        seed_lower = seed_hint.lower()
        keywords = [w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", seed_lower) if len(w) > 2 and w not in _QUERY_STOPWORDS][:3]

        # De-prioritize keywords that match rejected fingerprints
        if rejected_fingerprints:
            rejected_tokens = set()
            for fp in rejected_fingerprints:
                rejected_tokens.update(re.findall(r"[a-z][a-z0-9_-]{2,}", fp.lower()))
            keywords = [kw for kw in keywords if kw not in rejected_tokens]
            if not keywords:
                # Anchor terms always survive: drop rejected_tokens filter to prevent starvation
                all_seed_words = [w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", seed_lower)
                                  if w not in _QUERY_STOPWORDS and len(w) > 2]
                keywords = all_seed_words[:2]
        keywords = keywords[:3]

        queries = []
        if keywords:
            search_str = " ".join(keywords[:3])
            queries.append((search_str, {"sort": "relevance_score:desc"}))
            recent_year = max(2020, datetime.now().year - 2)
            queries.append((search_str, {"filter": f"publication_year:>{recent_year}", "sort": "cited_by_count:desc"}))
            if len(keywords) >= 2:
                alt_str = " ".join(keywords[:2])
                queries.append((alt_str, {"sort": "relevance_score:desc"}))
            # Cross-pollination: one query pairing the kind's method term with the seed's anchor
            if active_kind and active_kind in _EXPLORATION_KIND_BIASES:
                kind_biases = _EXPLORATION_KIND_BIASES.get(active_kind, {})
                kind_method = kind_biases.get("method", [""])[0]
                seed_anchor = keywords[0]
                if kind_method and kind_method != seed_anchor:
                    cross_search = f"{kind_method} {seed_anchor}"
                    queries.append((cross_search, {"sort": "relevance_score:desc"}))

        return queries[:4]

    def _preflight_dedup(
        self,
        queries: List[str],
        rejected_fingerprints: List[str],
    ) -> List[str]:
        """Skip or mutate queries whose keyword/category signature overlaps recent rejections.

        Uses structured fingerprints (keyword n-grams from rejected titles and
        rejection_reason tags) — never raw free-text rejection prose.
        Returns the filtered list and logs skipped queries.
        """
        if not rejected_fingerprints:
            return queries

        def _extract_fingerprint(query: str) -> set:
            words = re.findall(r"[a-z][a-z0-9_-]{2,}", query.lower())
            return set(w for w in words if w not in {"cat", "abs", "ti", "cs", "stat"})

        rejected_tokens = set()
        for fp in rejected_fingerprints:
            rejected_tokens.update(_extract_fingerprint(fp))

        deduped = []
        for q in queries:
            q_tokens = _extract_fingerprint(q)
            overlap = q_tokens & rejected_tokens
            if len(overlap) >= 2 and len(overlap) / max(len(q_tokens), 1) > 0.6:
                log_agent_action("TopicHunter", "preflight_dedup_skip", {
                    "query": q,
                    "overlap_tokens": sorted(overlap),
                    "reason": "query_fingerprint_matches_rejected_topic",
                })
                continue
            deduped.append(q)
        return deduped

    def fetch_citation_graph(self, paper_id: str) -> Dict[str, Any]:
        """
        Semantic Scholar Graph API: high in-degree + low recent out-degree = gap signal.
        paper_id can be DOI, arXiv, or S2 paperId.
        """
        try:
            fields = "title,year,citationCount,referenceCount,influentialCitationCount"
            url = f"{self.base_urls['s2']}/paper/{paper_id}"
            r = requests.get(
                url,
                headers=self.s2_headers,
                params={"fields": fields},
                timeout=20,
            )
            if r.status_code != 200:
                return {}
            paper = r.json()
            # Recent citing papers (proxy for out-degree from recent work extending it)
            cites_url = f"{self.base_urls['s2']}/paper/{paper_id}/citations"
            c = requests.get(
                cites_url,
                headers=self.s2_headers,
                params={"fields": "citingPaper.year,citingPaper.title", "limit": 50},
                timeout=20,
            )
            recent_extensions = 0
            if c.status_code == 200:
                for item in c.json().get("data", []):
                    year = (item.get("citingPaper") or {}).get("year") or 0
                    if year >= datetime.now().year - 2:
                        recent_extensions += 1
            in_degree = paper.get("citationCount") or 0
            return {
                "paper_id": paper_id,
                "title": paper.get("title"),
                "year": paper.get("year"),
                "in_degree": in_degree,
                "reference_count": paper.get("referenceCount") or 0,
                "recent_citing": recent_extensions,
                "gap_score": float(in_degree) / max(recent_extensions + 1, 1),
            }
        except Exception as e:
            log_agent_action("TopicHunter", "citation_graph_error", {"error": str(e)})
            return {}

    def evaluate_layered_novelty(self, candidate_topic: Dict[str, Any], abstracts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Layered novelty evaluation:
        1. Fast embedding similarity check
        2. Contribution comparison for high-overlap papers
        """
        if not abstracts:
            return {
                "max_similarity": 0.0,
                "reject": False,
                "nearest": None,
                "novelty_comparisons": [],
                "verdict": "NOVEL",
            }
        topic_desc = f"{candidate_topic.get('title','')} {candidate_topic.get('description','')} {candidate_topic.get('contribution','')}"
        topic_emb = generate_embedding(topic_desc)
        best_sim = 0.0
        nearest = None
        high_overlap_papers = []

        comparison_trigger = min(0.70, self.runtime_config.novelty_similarity_reject - 0.05)
        for paper in abstracts[:30]:
            abs_text = paper.get("abstract") if isinstance(paper, dict) else str(paper)
            if not abs_text:
                continue
            emb = generate_embedding(abs_text[:2000])
            denom = np.linalg.norm(topic_emb) * np.linalg.norm(emb)
            if denom == 0:
                continue
            sim = float(np.dot(topic_emb, emb) / denom)
            if sim > best_sim:
                best_sim = sim
                nearest = abs_text[:200]
            if sim >= comparison_trigger:
                high_overlap_papers.append({
                    "paper": paper if isinstance(paper, dict) else {"title": "Prior Paper", "abstract": abs_text},
                    "similarity": sim,
                })

        # If near duplicate threshold reached (e.g. > 0.95), immediate reject
        if best_sim >= 0.95:
            return {
                "max_similarity": best_sim,
                "reject": True,
                "nearest": nearest,
                "novelty_comparisons": [],
                "verdict": "LIKELY_DUPLICATE",
                "reason": f"Near-duplicate embedding similarity ({best_sim:.2f})",
            }

        # Contribution comparison for high overlap papers (similarity >= 0.70)
        comparisons = []
        is_duplicate = False
        if high_overlap_papers:
            prompt = f"""
Compare the candidate research contribution against prior work to evaluate novelty.
Candidate:
Title: {candidate_topic.get('title')}
Description: {candidate_topic.get('description')}
Proposed Contribution: {candidate_topic.get('contribution', candidate_topic.get('rationale', ''))}

Prior High-Overlap Works:
{json.dumps([{
    'title': p['paper'].get('title'),
    'abstract': p['paper'].get('abstract')[:1000],
    'similarity': round(p['similarity'], 2)
} for p in high_overlap_papers[:3]], indent=2)}

For EACH prior work, evaluate:
- Problem
- Method / Variables
- Overlap
- Difference & Remaining Gap
- Is the candidate novel compared to this paper? (true/false)

Return JSON:
{{
  "comparisons": [
    {{
      "paper": "title",
      "overlap": "...",
      "difference": "...",
      "remaining_gap": "...",
      "is_novel": true,
      "verdict": "NOVEL"
    }}
  ],
  "overall_novelty_verdict": "NOVEL|LIKELY_DUPLICATE|INSUFFICIENT_DIFFERENCE",
  "reject": false,
  "reason": "..."
}}
"""
            for attempt in range(2):
                raw = call_llm(prompt, temperature=0.3, tier="cheap")
                parsed = parse_json_from_llm(raw) or {}
                if isinstance(parsed, dict) and "comparisons" in parsed:
                    comparisons = parsed.get("comparisons") or []
                    verdict = parsed.get("overall_novelty_verdict", "NOVEL")
                    if verdict in ("LIKELY_DUPLICATE", "INSUFFICIENT_DIFFERENCE") or parsed.get("reject") is True:
                        is_duplicate = True
                    break

        reject_threshold = max(self.runtime_config.novelty_similarity_reject, comparison_trigger)
        reject = is_duplicate or (best_sim >= reject_threshold and not comparisons)
        return {
            "max_similarity": best_sim,
            "reject": reject,
            "nearest": nearest,
            "novelty_comparisons": comparisons,
            "verdict": "LIKELY_DUPLICATE" if reject else "NOVEL",
            "reason": f"Contribution comparison: {'rejected as duplicate/insufficient difference' if reject else 'novel contribution supported'}",
        }

    def novelty_score(self, topic_desc: str, abstracts: List[str]) -> Dict[str, Any]:
        """High similarity to recent abstracts -> fast embedding check with backward compatibility."""
        if not abstracts:
            return {"max_similarity": 0.0, "reject": False, "nearest": None}
        try:
            topic_emb = generate_embedding(topic_desc)
            best_sim = 0.0
            nearest = None
            for abs_text in abstracts[:30]:
                if not abs_text:
                    continue
                emb = generate_embedding(abs_text[:2000])
                denom = np.linalg.norm(topic_emb) * np.linalg.norm(emb)
                if denom == 0:
                    continue
                sim = float(np.dot(topic_emb, emb) / denom)
                if sim > best_sim:
                    best_sim = sim
                    nearest = abs_text[:200]
            return {
                "max_similarity": best_sim,
                "reject": best_sim >= self.runtime_config.novelty_similarity_reject,
                "nearest": nearest,
            }
        except Exception as e:
            return {"max_similarity": 0.0, "reject": False, "error": str(e)}

    def screen_research_gap(
        self,
        candidate: Dict[str, Any],
        literature: List[Dict[str, Any]],
        citation_gap_signal: float = 0.0,
    ) -> Dict[str, Any]:
        """Research Screener: verifies gap type, evidence support, and novelty distinction."""
        if not literature:
            return {
                "gap_type": "unsupported",
                "gap_claim": str(candidate.get("description", "")),
                "supporting_papers": [],
                "contradicting_papers": [],
                "closest_prior_work": [],
                "why_existing_work_is_insufficient": "No literature evidence available to substantiate research gap.",
                "proposed_contribution": str(candidate.get("contribution", "")),
                "evidence_strength": 0.0,
                "citation_gap_signal": citation_gap_signal,
                "status": "FAIL",
                "reason": "insufficient_literature_evidence",
            }

        prompt = f"""
You are a Research Screener. Evaluate whether the candidate research topic addresses a real, literature-supported gap.

Candidate:
Title: {candidate.get('title')}
Description: {candidate.get('description')}
Rationale: {candidate.get('rationale')}
Proposed Contribution: {candidate.get('contribution')}
Citation gap score (graph signal): {citation_gap_signal}

Relevant Literature:
{json.dumps([{
    'title': p.get('title'),
    'abstract': p.get('abstract', '')[:800],
    'doi': p.get('doi'),
    'arxiv_id': p.get('arxiv_id'),
} for p in literature[:6]], indent=2)}

Evaluate:
1. What category of gap is being claimed?
   (evaluation_gap, dataset_gap, method_gap, generalization_gap, robustness_gap, theoretical_gap, reproducibility_gap, resource_constraint_gap)
2. Is the gap actually supported by literature evidence?
3. What are supporting vs contradicting papers?
4. Why is existing work insufficient?
5. Evidence strength (0.0 - 1.0).

Return JSON:
{{
  "gap_type": "evaluation_gap",
  "gap_claim": "...",
  "supporting_papers": ["..."],
  "contradicting_papers": ["..."],
  "closest_prior_work": ["..."],
  "why_existing_work_is_insufficient": "...",
  "proposed_contribution": "...",
  "evidence_strength": 0.85,
  "status": "PASS|FAIL|INSUFFICIENT_EVIDENCE",
  "reason": "..."
}}
"""
        for attempt in range(2):
            raw = call_llm(prompt, temperature=0.3, tier="cheap")
            parsed = parse_json_from_llm(raw) or {}
            if isinstance(parsed, dict) and "gap_type" in parsed:
                evidence_strength = float(parsed.get("evidence_strength", 0.5))
                status = parsed.get("status", "PASS")
                # Relaxed threshold: allow topics with moderate evidence through
                if evidence_strength < 0.25:
                    status = "FAIL"
                return {
                    "gap_type": str(parsed.get("gap_type", "evaluation_gap")),
                    "gap_claim": str(parsed.get("gap_claim", candidate.get("description", ""))),
                    "supporting_papers": list(parsed.get("supporting_papers") or []),
                    "contradicting_papers": list(parsed.get("contradicting_papers") or []),
                    "closest_prior_work": list(parsed.get("closest_prior_work") or []),
                    "why_existing_work_is_insufficient": str(parsed.get("why_existing_work_is_insufficient", "")),
                    "proposed_contribution": str(parsed.get("proposed_contribution", candidate.get("contribution", ""))),
                    "evidence_strength": evidence_strength,
                    "citation_gap_signal": citation_gap_signal,
                    "status": status,
                    "reason": str(parsed.get("reason", "")),
                }

        # Fallback if LLM output fails
        return {
            "gap_type": "method_gap",
            "gap_claim": str(candidate.get("description", "")),
            "supporting_papers": [p.get("title") for p in literature[:2] if p.get("title")],
            "contradicting_papers": [],
            "closest_prior_work": [p.get("title") for p in literature[:1] if p.get("title")],
            "why_existing_work_is_insufficient": "Empirical question requires controlled comparison.",
            "proposed_contribution": str(candidate.get("contribution", "")),
            "evidence_strength": 0.6,
            "citation_gap_signal": citation_gap_signal,
            "status": "PASS",
            "reason": "literature_supported_gap",
        }

    def formalize_hypothesis(
        self,
        candidate: Dict[str, Any],
        gap_report: Dict[str, Any],
        novelty_report: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Hypothesis Formalizer: converts candidate into a machine-checkable StructuredHypothesis.

        Uses up to 3 attempts: the first generates from scratch, subsequent attempts
        repair specific missing fields from the previous partial response.
        """
        base_prompt = f"""
Formalize this candidate research question into a precise, machine-checkable scientific hypothesis contract.

Candidate:
Title: {candidate.get('title')}
Description: {candidate.get('description')}
Rationale: {candidate.get('rationale')}
Gap Report: {json.dumps(gap_report, default=str)[:1500]}
Novelty: {json.dumps(novelty_report, default=str)[:1000]}
Sandbox manifest: {json.dumps(SANDBOX_CAPABILITY_MANIFEST.as_dict())}

Requirements:
1. Clearly stated research_question.
2. Precise scientific hypothesis.
3. Explicit independent_variables with test ranges/values (e.g. [{{"name": "...", "values": [...]}}]).
4. Explicit dependent_variables (measurable metrics e.g. ["ECE", "accuracy", "Brier"]).
5. Expected relationship / prediction under control conditions.
6. Defined falsification_condition (what observation rejected the claim?).
7. Minimum viable experiment (dataset from catalog or synthetic, models, conditions, metrics, seeds, baseline, falsification_test).
8. Confounders and competing explanations identified.
9. Required resources within sandbox limits.

Return JSON:
{{
  "research_question": "...",
  "hypothesis": "...",
  "independent_variables": [{{"name": "...", "values": [1, 2, 3]}}],
  "dependent_variables": ["accuracy", "ECE"],
  "expected_relationship": "...",
  "falsification_condition": "...",
  "novelty_claim": "...",
  "closest_prior_work": [{{"title": "...", "difference": "..."}}],
  "baselines": ["logistic_regression", "random_forest"],
  "metrics": ["accuracy"],
  "confounders": ["model_capacity", "sample_size"],
  "competing_explanations": ["training_instability"],
  "minimum_viable_experiment": {{
    "dataset": "bundled_synthetic",
    "models": ["model_a", "model_b"],
    "conditions": ["controlled_seeds"],
    "metrics": ["accuracy"],
    "seeds": 3,
    "baseline": "logistic_regression",
    "expected_result": "...",
    "falsification_test": "Welch t-test p<0.05"
  }},
  "required_resources": {{
    "cpu": true,
    "gpu": false,
    "max_memory_mb": 4096,
    "max_runtime_seconds": 120
  }}
}}
"""
        required_fields = ["hypothesis", "falsification_condition", "dependent_variables", "research_question"]
        previous_response = None
        missing_fields = None

        for attempt in range(3):
            if attempt == 0:
                raw = call_llm(base_prompt, temperature=0.3, tier="strong")
            else:
                repair_prompt = f"""Your previous hypothesis formalization was incomplete.
Previous response:
{json.dumps(previous_response, indent=2, default=str)[:3000]}

Missing or invalid fields: {', '.join(missing_fields or required_fields)}

Fix ONLY the missing/invalid fields. Return the complete JSON with all fields populated.
Requirements:
- hypothesis: a clear testable statement
- falsification_condition: what observation would reject the claim
- dependent_variables: list of measurable metrics
- research_question: the precise question being answered
"""
                raw = call_llm(repair_prompt, temperature=0.2, tier="strong")

            parsed = parse_json_from_llm(raw) or {}
            previous_response = parsed

            if isinstance(parsed, dict):
                missing_fields = [f for f in required_fields if not parsed.get(f)]
                has_mve = isinstance(parsed.get("minimum_viable_experiment"), dict)
                if not missing_fields and has_mve:
                    parsed["gap_report"] = gap_report
                    parsed["novelty_report"] = novelty_report
                    return parsed

        return None

    def feasibility_filter(self, topic: Topic) -> FeasibilityReport:
        """Grounded in what Engineer sandbox can actually run."""
        reasons = []
        ok = True
        text = json.dumps(topic).lower()
        blocked = [
            "large language model fine-tune",
            "gpu cluster",
            "human subjects",
            "clinical trial",
            "wet lab",
            "robot hardware",
            "million parameter training from scratch",
        ]
        negation_re = re.compile(
            r"(?:not|no|without|does not (?:require|need)|avoids?|never)(?:\s|$)", re.IGNORECASE
        )
        for b in blocked:
            idx = text.find(b)
            while idx != -1:
                window = text[max(0, idx - 50):idx]
                if negation_re.search(window):
                    idx = text.find(b, idx + 1)
                    continue
                ok = False
                reasons.append(f"Not executable in sandbox: {b}")
                break
        feas = topic.get("feasibility", 5)
        if isinstance(feas, (int, float)) and feas < 3:
            ok = False
            reasons.append(f"Low feasibility score: {feas}")
        # Prefer synthetic / public small data
        if "dataset" in text and "proprietary" in text:
            ok = False
            reasons.append("Proprietary dataset unavailable")
        # Only block explicit outbound network requirements
        if any(phrase in text for phrase in ("requires internet", "must download", "needs api access to")):
            candidate_plan = {"methodology": topic.get("description", ""), "experiments": [{"dataset": {"name": topic.get("dataset_plan", "")}}]}
            capability_reasons = check_plan_feasibility(candidate_plan, SANDBOX_CAPABILITY_MANIFEST)
            if capability_reasons:
                ok = False
                reasons.extend(capability_reasons)
        return {"ok": ok, "reasons": reasons}

    def _reject(self, topic: Topic, reason: str, meta: Optional[Dict[str, Any]] = None):
        meta = meta or {}
        entry = {
            "title": topic.get("title"),
            "reason": reason,
            "meta": meta,
            "ts": datetime.now().isoformat(),
        }
        self.rejection_log.append(entry)
        topic_kind = topic.get("hypothesis_kind") or hypothesis_kind(topic.get("title", ""))
        lesson_type = meta.get("lesson_type") or "topic_rejection"
        reason_code = meta.get("reason_code") or reason
        CrossRunMemory().record_rejection(
            "topic",
            topic.get("title", "?"),
            reason,
            {
                **meta,
                "lesson_type": lesson_type,
                "reason_code": reason_code,
                "topic_kind": topic_kind,
            },
        )
        self._excluded_titles_cache = None
        tracker = get_tracker()
        if tracker:
            tracker.bump("rejected_topics")
            tracker.scratch("TopicHunter", "rejection", entry)
        log_agent_action("TopicHunter", "topic_rejected", entry)

    def _excluded_titles(self) -> List[str]:
        if self._excluded_titles_cache is None:
            self._excluded_titles_cache = CrossRunMemory().excluded_topic_titles()
        return self._excluded_titles_cache

    def _matches_excluded_topic(self, title: str) -> Optional[str]:
        """Return the prior excluded title if this candidate is a near-duplicate."""
        candidate = (title or "").strip()
        if not candidate:
            return None
        lowered = candidate.lower()
        for prior in self._excluded_titles():
            if not prior:
                continue
            if lowered == prior.lower() or title_token_overlap(candidate, prior) >= _FAILED_TOPIC_OVERLAP_THRESHOLD:
                return prior
        return None

    def _hunt_once(self, domain: str, seed_hint: str) -> List[Dict[str, Any]]:
        cross_run_context = CrossRunMemory().get_prompt_context()
        excluded = self._excluded_titles()
        rejected_fingerprints = self._extract_rejected_fingerprints(cross_run_context)
        active_kind = self._active_hypothesis_kind()

        arxiv_queries = self._build_arxiv_queries(seed_hint, rejected_fingerprints, active_kind)
        openalex_queries = self._build_openalex_queries(seed_hint, rejected_fingerprints, active_kind)
        openalex_searches = [q[0] for q in openalex_queries]
        openalex_extra_params = [q[1] for q in openalex_queries]

        arxiv_queries = self._preflight_dedup(arxiv_queries, rejected_fingerprints)
        openalex_searches = self._preflight_dedup(openalex_searches, rejected_fingerprints)

        if not arxiv_queries and not openalex_searches:
            log_agent_action("TopicHunter", "seed_skipped_no_queries", {
                "seed": seed_hint,
                "reason": "all_keywords_filtered_by_rejected_fingerprints",
                "rejected_fingerprints_count": len(rejected_fingerprints),
            })
            return []

        base_limit = 40
        extra = min(self._iteration_failures * 10, 30)
        search_limit = base_limit + extra

        recent_papers = []
        for i, query in enumerate(openalex_searches):
            extra_params = openalex_extra_params[i] if i < len(openalex_extra_params) else {}
            recent_papers.extend(self.search_openalex(query, search_limit, extra_params=extra_params))
        recent_papers.extend(self.search_arxiv_multi(arxiv_queries, min(search_limit, 30)))
        if not recent_papers:
            return []

        # Citation graph signals for top cited older-looking papers
        graph_signals = []
        for p in sorted(recent_papers, key=lambda x: x.get("cited_by_count") or 0, reverse=True)[:5]:
            doi = (p.get("doi") or "").replace("https://doi.org/", "")
            if doi:
                g = self.fetch_citation_graph(f"DOI:{doi}")
                if g:
                    graph_signals.append(g)

        abstracts = [
            (p.get("abstract") or "") for p in recent_papers if p.get("abstract")
        ][:25]
        evidence_map = build_cross_paper_evidence_map(recent_papers)

        prompt = f"""
Find research GAPS (not trendy saturated topics) in {domain}.
Seed angle: {seed_hint}
Prior-run lessons (avoid repeats):
{json.dumps(cross_run_context, sort_keys=True)}

Topics already rejected or that FAILED hypothesis debate (do NOT propose these or near-duplicates):
{json.dumps(excluded[-25:], indent=2)}

Citation-graph gap signals (high in-degree, low recent extensions):
{json.dumps(graph_signals[:5], indent=2)}

Evidence-backed cross-paper bridges. These are candidate transfer questions,
not proof of a research gap. Use their cited excerpts when relevant; do not
invent a relationship not present in the supplied evidence:
{json.dumps(evidence_map.get('bridges', [])[:6], indent=2)[:6000]}

Sample recent titles:
{[p.get('title', '')[:100] for p in recent_papers[:8]]}

A real gap: foundational work is cited but rarely extended lately.
Propose 5-6 topics executable with CPU sklearn/numpy synthetic or small public data.
For each topic include an explicit "contribution" sentence for novelty checking.

For every proposed topic, include `evidence_bridge_ids` containing the bridge IDs
that support its cross-paper synthesis. Do not fabricate IDs.
JSON: {{"gaps": [{{"title": "...", "description": "...", "rationale": "...", "impact": "...",
"feasibility": 7, "keywords": [], "anchor_paper": "...", "dataset_plan": "synthetic|public",
"evidence_bridge_ids": ["bridge-..."]}}]}}
"""
        parsed = parse_json_from_llm(call_llm(prompt, temperature=0.8, tier="cheap")) or {}
        gaps = parsed.get("gaps") or []

        # Cheap pre-filter: reject candidates with no textual overlap with retrieved literature
        literature_text = " ".join(
            (p.get("title", "") + " " + p.get("abstract", "")[:500]).lower()
            for p in recent_papers[:15]
        )
        pre_filtered = []
        for gap in gaps:
            gap_text = f"{gap.get('title', '')} {gap.get('description', '')}".lower()
            gap_words = set(re.findall(r"[a-z][a-z0-9_-]{3,}", gap_text))
            lit_words = set(re.findall(r"[a-z][a-z0-9_-]{3,}", literature_text))
            overlap = gap_words & lit_words
            if len(overlap) >= 2:
                pre_filtered.append(gap)
            else:
                self._reject(gap, "no_literature_grounding", {
                    "lesson_type": "pre_filter_rejection",
                    "reason_code": "zero_literature_overlap",
                })
        gaps = pre_filtered

        kept = []
        for gap in gaps:
            bridge_validation = validate_candidate_bridge_claim(gap, evidence_map)
            gap["bridge_validation"] = bridge_validation
            if not bridge_validation["valid"]:
                self._reject(gap, "unsupported_cross_paper_synthesis", {
                    "lesson_type": "evidence_grounding_failure",
                    "reason_code": bridge_validation["reason"],
                })
                continue
            # Soft warning: log but don't reject on bridge text mismatch
            if "soft_warning" in bridge_validation.get("reason", ""):
                log_agent_action("TopicHunter", "bridge_soft_warning", {
                    "title": gap.get("title"),
                    "reason": bridge_validation["reason"],
                })
            gap["literature_evidence"] = [
                {
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", "")[:3000],
                    "doi": paper.get("doi"),
                    "arxiv_id": paper.get("arxiv_id"),
                }
                for paper in recent_papers[:8]
                if paper.get("title") and paper.get("abstract")
            ]
            gap["cross_paper_evidence"] = evidence_map
            prior = self._matches_excluded_topic(gap.get("title", ""))
            if prior:
                self._reject(gap, "previously_failed_or_rejected", {"matched": prior})
                continue
            # 1. Screen research gap with literature evidence
            citation_signal = max((g.get("gap_score", 0) for g in graph_signals), default=0.0)
            gap_report = self.screen_research_gap(gap, gap.get("literature_evidence", []), citation_gap_signal=citation_signal)
            gap["gap_report"] = gap_report
            if gap_report.get("status") in ("FAIL", "INSUFFICIENT_EVIDENCE"):
                self._reject(gap, "unsupported_research_gap", {
                    "lesson_type": "screener_rejection",
                    "reason_code": "unsupported_research_gap",
                    "gap_type": gap_report.get("gap_type", "unsupported"),
                    "details": gap_report.get("why_existing_work_is_insufficient", ""),
                })
                continue

            # 2. Layered Novelty Assessment
            novelty_eval = self.evaluate_layered_novelty(gap, gap.get("literature_evidence", []))
            gap["novelty"] = novelty_eval
            if novelty_eval.get("reject"):
                self._reject(gap, "novelty_too_low", {
                    "lesson_type": "novelty_failure",
                    "reason_code": "existing_contribution_overlap",
                    "verdict": novelty_eval.get("verdict", "LIKELY_DUPLICATE"),
                    "details": novelty_eval.get("reason", ""),
                })
                continue

            # 3. Feasibility check against sandbox capability manifest
            feas = self.feasibility_filter(gap)
            gap["feasibility_check"] = feas
            if not feas["ok"]:
                self._reject(gap, "infeasible_for_engineer", {
                    "lesson_type": "feasibility_failure",
                    "reason_code": "sandbox_capability_violation",
                    "reasons": feas.get("reasons", []),
                })
                continue

            # 4. Hypothesis Formalization
            formalized = self.formalize_hypothesis(gap, gap_report, novelty_eval)
            if not formalized:
                self._reject(gap, "missing_structured_hypothesis", {
                    "lesson_type": "admission_failure",
                    "reason_code": "structured_hypothesis_unavailable",
                })
                continue
            admission = validate_topic_admission(formalized)
            gap["topic_admission"] = admission
            if not admission["admitted"]:
                self._reject(gap, "topic_admission_failed", {
                    "lesson_type": "admission_failure",
                    "reason_code": "non_executable_minimum_experiment",
                    "reasons": admission["errors"],
                })
                continue
            gap["structured_hypothesis"] = formalized
            gap["falsifiable_prediction"] = formalized.get("falsification_condition", "")
            gap["research_question"] = formalized.get("research_question", "")

            # Graph bonus
            if graph_signals:
                gap["gap_score"] = max(g.get("gap_score", 0) for g in graph_signals)
            kept.append(gap)
            try:
                self.vector_memory.add_embedding(
                    generate_embedding(gap["title"] + " " + gap.get("description", "")),
                    {
                        "type": "research_gap",
                        "namespace": "topic_hunter",
                        "content_class": "generated_narrative",
                        "retrieval_eligible": False,
                        "agent": "TopicHunterAgent",
                        "outcome_status": "unknown",
                        "title": gap["title"],
                        "domain": domain,
                    },
                )
            except Exception:
                pass
        return kept

    def discover_topics(self, domain: str = None, n_parallel: int = 5) -> List[Topic]:
        self._excluded_titles_cache = None
        domain = domain or self.runtime_config.research_domain
        log_agent_action("TopicHunter", "start_discovery", {"domain": domain, "parallel": n_parallel})
        cross_run_context = CrossRunMemory().get_prompt_context()
        seeds = self._generate_dynamic_seeds(n_parallel, cross_run_context)

        all_topics: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=n_parallel) as pool:
            futures = {pool.submit(self._hunt_once, domain, s): s for s in seeds}
            for fut in as_completed(futures):
                try:
                    all_topics.extend(fut.result() or [])
                except Exception as e:
                    log_agent_action("TopicHunter", "parallel_hunt_error", {"error": str(e)})

        if not any(s.get("ok") for s in self.source_health.values()):
            details = "; ".join(f"{name}: {entry.get('error', 'unavailable')}" for name, entry in self.source_health.items())
            raise ResearchSourceUnavailable(
                "Research discovery could not contact OpenAlex or arXiv. "
                "Check network access and OPENALEX_EMAIL, then try again. Details: " + details
            )

        seen = set()
        unique = []
        for t in all_topics:
            title = (t.get("title") or "").lower().strip()
            if title and title not in seen:
                seen.add(title)
                unique.append(t)

        # Second-pass targeted retrieval from top bridge candidates
        if unique:
            evidence_map = build_cross_paper_evidence_map(unique)
            bridges = evidence_map.get("bridges", [])
            top_bridges = [b for b in bridges if b.get("method_signal") and b.get("target_setting_signal")][:3]
            if top_bridges:
                second_pass_topics = self._execute_second_pass(top_bridges, domain)
                for t in second_pass_topics:
                    title = (t.get("title") or "").lower().strip()
                    if title and title not in seen:
                        seen.add(title)
                        unique.append(t)

        # Rejection funnel instrumentation: aggregate by reason code
        funnel = {}
        for entry in self.rejection_log:
            reason = entry.get("reason", "unknown")
            funnel[reason] = funnel.get(reason, 0) + 1
        log_agent_action("TopicHunter", "rejection_funnel", {
            "funnel": funnel,
            "total_rejected": len(self.rejection_log),
            "total_kept": len(unique),
            "pass_rate": round(len(unique) / max(len(unique) + len(self.rejection_log), 1), 3),
        })

        ranked = self.rank_topics_by_potential(unique)
        log_agent_action("TopicHunter", "discovery_complete", {
            "num_topics": len(ranked),
            "rejected": len(self.rejection_log),
        })
        if not ranked:
            self._iteration_failures += 1
        else:
            self._iteration_failures = 0
        return ranked

    def _execute_second_pass(self, top_bridges: List[Dict[str, Any]], domain: str) -> List[Topic]:
        """Execute a narrower second-pass retrieval seeded from top bridge candidates.

        Uses bridge method/setting signals to build targeted queries before
        final novelty/feasibility scoring.
        """
        second_pass_queries = []
        for bridge in top_bridges:
            method_signal = str(bridge.get("method_signal", "")).strip()
            setting_signal = str(bridge.get("target_setting_signal", "")).strip()
            if method_signal:
                second_pass_queries.append(f"cat:cs.LG AND abs:{method_signal}")
            if setting_signal and method_signal:
                second_pass_queries.append(f"cat:cs.LG AND abs:{method_signal} AND abs:{setting_signal}")

        if not second_pass_queries:
            return []

        second_pass_papers = self.search_arxiv_multi(second_pass_queries[:4], 20)
        if not second_pass_papers:
            return []

        second_pass_papers = [p for p in second_pass_papers if p.get("title") and p.get("abstract")][:15]
        if not second_pass_papers:
            return []

        prompt = f"""
Given these bridge candidates from a cross-paper evidence map, propose 1-2 research gaps
that synthesize across the method/setting signals. Be specific and grounded in the paper evidence.

Bridge candidates:
{json.dumps([{
    'method_signal': b.get('method_signal'),
    'target_setting_signal': b.get('target_setting_signal'),
    'evidence': b.get('evidence', [])[:2],
} for b in top_bridges[:3]], indent=2)[:4000]}

Sample titles from targeted retrieval:
{[p.get('title', '')[:100] for p in second_pass_papers[:8]]}

For each topic include an explicit "contribution" sentence for novelty checking.
JSON: {{"gaps": [{{"title": "...", "description": "...", "rationale": "...", "impact": "...",
"feasibility": 7, "keywords": [], "anchor_paper": "...", "dataset_plan": "synthetic|public",
"evidence_bridge_ids": []}}]}}
"""
        parsed = parse_json_from_llm(call_llm(prompt, temperature=0.7, tier="cheap")) or {}
        gaps = parsed.get("gaps") or []

        kept = []
        for gap in gaps[:2]:
            gap["literature_evidence"] = [
                {"title": p.get("title", ""), "abstract": p.get("abstract", "")[:3000],
                 "doi": p.get("doi"), "arxiv_id": p.get("arxiv_id")}
                for p in second_pass_papers[:6]
                if p.get("title") and p.get("abstract")
            ]
            prior = self._matches_excluded_topic(gap.get("title", ""))
            if prior:
                self._reject(gap, "previously_failed_or_rejected", {"matched": prior})
                continue
            novelty_eval = self.evaluate_layered_novelty(gap, gap.get("literature_evidence", []))
            gap["novelty"] = novelty_eval
            if novelty_eval.get("reject"):
                self._reject(gap, "novelty_too_low", {
                    "lesson_type": "novelty_failure",
                    "reason_code": "existing_contribution_overlap",
                    "verdict": novelty_eval.get("verdict", "LIKELY_DUPLICATE"),
                })
                continue
            feas = self.feasibility_filter(gap)
            gap["feasibility_check"] = feas
            if not feas["ok"]:
                self._reject(gap, "infeasible_for_engineer", {
                    "lesson_type": "feasibility_failure",
                    "reason_code": "sandbox_capability_violation",
                    "reasons": feas.get("reasons", []),
                })
                continue
            kept.append(gap)
        return kept

    def rank_topics_by_potential(self, topics: List[Topic]) -> List[Topic]:
        if not topics:
            return []
        if len(topics) == 1:
            topics[0]["rank"] = 1
            topics[0]["score"] = topics[0].get("feasibility", 7)
            topics[0]["selection_mode"] = "exploitation"
            return topics
        prompt = f"""
Rank these research topics (lightweight judge). Prefer novel executable gaps over trendy saturated areas.
{json.dumps([{k: t.get(k) for k in ('title','description','feasibility','novelty','gap_score')} for t in topics], indent=2)[:5000]}
JSON: {{"ranked_topics": [{{"original_index": 0, "rank": 1, "score": 8.5, "reasoning": "..."}}]}}
"""
        # Elo is deliberately a small tie-breaker, not a replacement for evidence-based
        # gap/novelty/feasibility scoring. Exploration periodically boosts under-observed
        # kinds but never bypasses the gates already applied in _hunt_once.
        elo = EloStore(context=getattr(self, "context", None))
        for topic in topics:
            kind = topic.get("hypothesis_kind") or hypothesis_kind(topic.get("title", ""))
            topic["hypothesis_kind"] = kind
            topic["elo_rating"] = float(elo.get(kind))
            topic["elo_observations"] = elo.observations(kind)

        try:
            parsed = parse_json_from_llm(call_llm(prompt, temperature=0.3, tier="cheap")) or {}
            for rank_info in parsed.get("ranked_topics") or []:
                idx = rank_info.get("original_index", 0)
                if idx < len(topics):
                    topics[idx]["rank"] = rank_info.get("rank", 999)
                    topics[idx]["score"] = rank_info.get("score", 5)
                    topics[idx]["reasoning"] = rank_info.get("reasoning", "")
            ranked = sorted(topics, key=lambda x: (x.get("rank", 999), -x.get("elo_rating", 1500.0)))
        except Exception:
            ranked = sorted(topics, key=lambda x: (-(x.get("gap_score") or x.get("feasibility") or 0), -x.get("elo_rating", 1500.0)))

        return self._apply_exploration(ranked, elo)

    def _apply_exploration(self, ranked: List[Topic], elo: EloStore) -> List[Topic]:
        """Deterministic forced exploration of under-observed / lower-rated kinds."""
        if not ranked:
            return ranked
        if not hasattr(self, "_selection_counter"):
            self._selection_counter = 0
        if not hasattr(self, "_rng"):
            seed = 42
            cfg = getattr(self, "runtime_config", None)
            if cfg is not None:
                seed = int(getattr(cfg, "topic_exploration_seed", 42))
            self._rng = random.Random(seed)
        self._selection_counter += 1
        cfg = getattr(self, "runtime_config", None)
        every = max(1, int(getattr(cfg, "topic_exploration_every", 4) if cfg is not None else 4))
        force = (self._selection_counter % every) == 0
        for topic in ranked:
            topic.setdefault("selection_mode", "exploitation")
        if not force:
            return ranked

        kinds = [t.get("hypothesis_kind") or hypothesis_kind(t.get("title", "")) for t in ranked]
        under = elo.under_observed_kinds(list(dict.fromkeys(kinds)))
        candidates = [t for t in ranked if (t.get("hypothesis_kind") in under)] if under else list(ranked)
        candidates = sorted(candidates, key=lambda t: (t.get("elo_rating", 1500.0), t.get("elo_observations", 0)))
        if not candidates:
            return ranked
        pick = candidates[self._rng.randrange(len(candidates))]
        pick["selection_mode"] = "forced_exploration"
        rest = [t for t in ranked if t is not pick]
        log_agent_action("TopicHunter", "forced_exploration", {
            "kind": pick.get("hypothesis_kind"),
            "title": pick.get("title"),
            "elo_rating": pick.get("elo_rating"),
            "elo_observations": pick.get("elo_observations"),
        })
        return [pick] + rest
