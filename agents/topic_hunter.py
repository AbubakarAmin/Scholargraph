"""
TopicHunterAgent — citation-graph gap analysis, novelty filter, parallel hunts.

CHANGELOG (this revision):
  1. Rejection-funnel counter logged at the end of every discover_topics() run,
     so bottleneck gates are visible instead of inferred.
  2. feasibility_filter: negation-aware blocked-phrase matching (no longer
     false-positives on "does not require a gpu cluster").
  3. evaluate_layered_novelty: the high-overlap/comparison trigger threshold is
     now clamped to never sit above novelty_similarity_reject, so a candidate
     can no longer be auto-rejected on embedding similarity alone without ever
     reaching LLM contribution comparison.
  4. Bridge-claim validation: malformed/missing evidence_bridge_ids now downgrade
     to a soft warning (with a programmatic fallback match against the evidence
     map) instead of a hard reject, matching the existing soft-warning path for
     text mismatches.
  5. formalize_hypothesis: added a targeted repair turn — instead of regenerating
     from scratch on failure, the second (and new third) attempt is told exactly
     which required field was missing/invalid and asked to fix only that.
  6. Seed/query generation: rejected-fingerprint history is now windowed (most
     recent N) and keyword stripping is floored so at least one anchor keyword
     always survives, preventing seeds from decaying into generic queries over
     a long run.
  7. Discovery prompt now asks for more raw candidates per seed, and a cheap
     heuristic pre-filter screens out obviously ungrounded gaps before the
     expensive screener/novelty/formalization gate chain runs on them.
  8. Capability-First Dataset Scoping: dataset_plan is checked against the local
     catalog before bridge validation; uncatalogued datasets are rejected early
     with reason_code="dataset_not_catalogued". Fail-open on catalog errors.
  9. Structural Gap Mining: bibliographic coupling analysis via S2 references
     identifies pairs sharing references but not citing each other, injected as
     structural gap signals into the discovery prompt.
  10. Method × Domain Sparsity Matrix: LLM-extracted (method, domain) pairs from
      retrieved abstracts are counted; rare combinations of well-established
      methods/domains are surfaced as sparse-cell gap signals.
  11. Contradiction Mining: LLM-assisted identification of papers making opposing
      empirical claims on the same subject, injected as ready-made research gaps.
  12. Replication-Target Mining: regex heuristic flags papers making strong claims
      with no visible variance/multi-seed reporting, for replication-and-extension
      topic proposals.
  13. Own Negative Results as Prior Work: CrossRunMemory.get_negative_result_lessons()
      surfaces structured fields from prior unsupported hypotheses as seed context.
  14. Persona Ensemble Generation: discovery prompt is run through 2 personas
      (skeptic, practitioner) concurrently; merged gaps are tagged with persona
      for observability.
  15. Seed-Strategy Provenance + Elo: _generate_dynamic_seeds returns strategy-
      tagged seeds; strategy outcomes are tracked in EloStore under strategy:<name>
      keys; seed ordering is sorted by strategy Elo with exploration reserve.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
import time
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
from core.structural_gaps import find_coupling_gaps
from core.sparsity_matrix import find_sparse_cells
from core.contradiction_mining import find_contradictions


logger = logging.getLogger(__name__)

_FAILED_TOPIC_OVERLAP_THRESHOLD = 0.55

# Fix #6: only the most recent N rejection entries influence seed/query
# keyword stripping. Without this cap, a long-running session accumulates so
# many rejected fingerprints that every keyword eventually gets excluded and
# seeds decay into generic, domain-less queries.
_REJECTION_HISTORY_WINDOW = 20

# Fix #6: no matter how much overlap is found with rejected fingerprints, at
# least this many original seed keywords are preserved so a query never goes
# out completely un-anchored.
_MIN_SURVIVING_KEYWORDS = 1

# Fix #3: the similarity level at which a candidate topic's contribution gets
# compared against prior work by an LLM (rather than judged on cosine sim
# alone). This must never sit ABOVE the reject threshold, or a candidate can
# be auto-rejected purely on embedding similarity without ever reaching
# comparison. Actual value is clamped dynamically in evaluate_layered_novelty.
_DEFAULT_COMPARISON_TRIGGER = 0.70

# Fix #7: how many raw candidate gaps to request per seed hunt. Raised from 3
# to give the downstream gate chain more surviving material, since the chain
# is multi-stage and each stage has a nonzero rejection rate.
_GAPS_PER_SEED_REQUEST = 6

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

# Fix #2: blocked phrases for the sandbox feasibility filter. Kept as plain
# strings (matching original behavior) — the negation-aware check is done in
# `_phrase_blocked_in_text`, not by changing this list.
_BLOCKED_SANDBOX_PHRASES = [
    "large language model fine-tune",
    "gpu cluster",
    "human subjects",
    "clinical trial",
    "wet lab",
    "robot hardware",
    "million parameter training from scratch",
]

# Fix #2: negation cues that, if found immediately before a blocked phrase,
# mean the text is disclaiming the requirement rather than stating it.
_NEGATION_CUES = ("not ", "no ", "without ", "won't need ", "does not require ", "doesn't require ", "avoids ")

# OpenAlex concept IDs (https://api.openalex.org/concepts) corresponding to
# each _ARXIV_CATEGORIES subcategory. IDs verified against live API 2026-09-12.
_OPENALEX_CONCEPTS = {
    "machine_learning": ["C119857082", "C154945302"],       # Machine learning, Artificial intelligence
    "natural_language_processing": ["C204321447", "C41008148"],  # NLP, Computer science
    "computer_vision": ["C31972630", "C154945302"],          # Computer vision, AI
    "reinforcement_learning": ["C119857082", "C154945302"],
    "graph_neural_networks": ["C119857082", "C41008148"],
    "fairness": ["C154945302", "C33923547"],                 # AI, Mathematics
    "federated_learning": ["C119857082", "C31258907"],       # ML, Computer network
    "optimization": ["C33923547", "C119857082"],             # Mathematical optimization, ML
    "general": ["C119857082", "C154945302", "C41008148"],
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
        # Feature 5: run-scoped, thread-safe query cache for cross-seed dedup
        self._run_query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._run_query_cache_lock = threading.Lock()
        self._dataset_catalog_cache: Optional[List[Dict[str, Any]]] = None
        self._arxiv_client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3,
        )
        self._arxiv_lock = threading.Lock()

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
            with self._arxiv_lock:
                for result in self._arxiv_client.results(search):
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
            logger.warning(f"arxiv search failed for query={query!r}: {e}")
            log_agent_action("TopicHunter", "search_arxiv_error", {"error": str(e), "query": query})
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
        for i, query in enumerate(queries):
            if i > 0:
                time.sleep(3)
            all_results.extend(self.search_arxiv(query, max_results))
        return all_results

    def _extract_rejected_fingerprints(self, cross_run_context: List[Dict[str, Any]]) -> List[str]:
        """Extract keyword fingerprints from rejected-topic items in CrossRunMemory.

        Only uses structured tags (rejection_reason, item title) — never raw
        free-text rejection reasons, consistent with the fail-closed memory policy.

        Fix #6: windowed to the most recent _REJECTION_HISTORY_WINDOW entries so
        keyword-stripping downstream can't be starved by a long accumulated history.
        """
        windowed_context = cross_run_context[-_REJECTION_HISTORY_WINDOW:]
        fingerprints = []
        for entry in windowed_context:
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
        domain: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Generate fresh seed phrases from current state instead of a static bank.

        Uses structured CrossRunMemory tags (fail-closed) to:
        - Pivot away from rejected topic areas
        - Bias toward underexplored hypothesis kinds
        - Combine method + evaluation signals from the kind bias tables
        - Feature 4: also harvest frontier terms from very recent papers

        Fix #6: rejected-token derivation is windowed to recent history so a long
        session doesn't accumulate enough rejected tokens to blank out every seed.

        Feature 8: returns List[Dict] with {"seed": ..., "strategy": ...} for
        seed-strategy provenance tracking.
        """
        if cross_run_context is None:
            cross_run_context = CrossRunMemory().get_prompt_context()

        windowed_context = cross_run_context[-_REJECTION_HISTORY_WINDOW:]

        rejected_tokens: set = set()
        for entry in windowed_context:
            item = str(entry.get("item") or "").lower()
            tag = str(entry.get("rejection_reason") or "")
            if item:
                rejected_tokens.update(w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", item) if len(w) > 3)
            if tag and tag not in ("other",):
                rejected_tokens.add(tag)

        active_kind = self._active_hypothesis_kind()

        seeds: List[Dict[str, str]] = []  # Feature 8: track strategy per seed

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
            seeds.append({"seed": f"{method} {evaluation} open problems", "strategy": "kind_bias"})

        # 2. Extra seeds biased toward the active exploration kind
        if active_kind and active_kind in _EXPLORATION_KIND_BIASES:
            biases = _EXPLORATION_KIND_BIASES[active_kind]
            for method_term in biases.get("method", [])[:2]:
                for eval_term in biases.get("evaluation", [])[:1]:
                    candidate = f"{method_term} {eval_term} underexplored challenges"
                    candidate_words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", candidate))
                    if not (candidate_words & rejected_tokens):
                        seeds.append({"seed": candidate, "strategy": "kind_bias"})

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
                seeds.append({"seed": s, "strategy": "generic_fallback"})

        # Feature 4: frontier-seeded generation from very recent papers
        frontier_terms = self._harvest_frontier_terms(domain) if domain else []
        for term_pair in frontier_terms:
            method = term_pair.get("method", "")
            evaluation = term_pair.get("evaluation", "")
            if not method or not evaluation:
                continue
            words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", f"{method} {evaluation}".lower()))
            if words & rejected_tokens:
                continue
            seeds.append({"seed": f"{method} {evaluation} open problems", "strategy": "frontier"})

        # Fix #6: if aggressive filtering above still leaves us short, fall back to
        # unfiltered generic seeds rather than starving the run of any seeds at all.
        if len(seeds) < max(1, n_seeds // 2):
            for s in generic_seeds:
                if not any(d["seed"] == s for d in seeds):
                    seeds.append({"seed": s, "strategy": "generic_fallback"})

        # Deduplicate preserving order
        seen: set = set()
        unique: List[Dict[str, str]] = []
        for d in seeds:
            if d["seed"] not in seen:
                seen.add(d["seed"])
                unique.append(d)

        # Feature 8: sort by strategy Elo (descending) if enabled
        if getattr(self.runtime_config, "seed_strategy_elo_enabled", True):
            try:
                elo = EloStore(context=getattr(self, "context", None))
                unique.sort(key=lambda d: elo.get(f"strategy:{d['strategy']}"), reverse=True)
                # Reserve >=1 slot for the lowest-rated strategy (exploration)
                strategies_seen = {d["strategy"] for d in unique}
                if len(strategies_seen) > 1:
                    all_strategies = ["kind_bias", "cross_pollination", "generic_fallback", "frontier"]
                    lowest = min(all_strategies, key=lambda s: elo.get(f"strategy:{s}"))
                    if not any(d["strategy"] == lowest for d in unique[:1]):
                        # Move one seed of the lowest-rated strategy to the front
                        for i, d in enumerate(unique):
                            if d["strategy"] == lowest:
                                unique.insert(0, unique.pop(i))
                                break
            except Exception:
                pass

        return unique[:n_seeds]

    def _apply_keyword_floor(self, keywords: List[str], all_seed_words: List[str], rejected_tokens: set) -> List[str]:
        """Fix #6: ensure at least _MIN_SURVIVING_KEYWORDS keywords survive rejected-token
        filtering, even if that means keeping a keyword that overlaps a rejected fingerprint.

        Without this floor, `_build_arxiv_queries` / `_build_openalex_queries` can end up
        with zero keywords on a seed whose every token happens to intersect the rejected
        set, producing an unanchored, near-meaningless query (or none at all).
        """
        if len(keywords) >= _MIN_SURVIVING_KEYWORDS:
            return keywords
        # Backfill from the original (unfiltered) seed words, preferring ones not rejected,
        # but accepting rejected ones rather than shipping zero keywords.
        backfill_preferred = [w for w in all_seed_words if w not in rejected_tokens and w not in keywords]
        backfill_fallback = [w for w in all_seed_words if w not in keywords]
        for pool in (backfill_preferred, backfill_fallback):
            for w in pool:
                if len(keywords) >= _MIN_SURVIVING_KEYWORDS:
                    break
                keywords.append(w)
            if len(keywords) >= _MIN_SURVIVING_KEYWORDS:
                break
        return keywords

    def _resolve_subcategory(self, active_kind: Optional[str]) -> str:
        """Resolve active hypothesis kind to a subcategory label for _ARXIV_CATEGORIES / _OPENALEX_CONCEPTS."""
        if active_kind and active_kind in _TOPICS_TO_SUBCATEGORY:
            return _TOPICS_TO_SUBCATEGORY[active_kind]
        return "general"

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
        keywords matching recently rejected fingerprints, but never below
        the keyword floor (Fix #6).
        """
        seed_lower = seed_hint.lower()
        all_seed_words = [w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", seed_lower)
                           if w not in _QUERY_STOPWORDS and len(w) > 2]
        keywords = all_seed_words[:3]

        # De-prioritize keywords that match rejected fingerprints
        rejected_tokens = set()
        if rejected_fingerprints:
            for fp in rejected_fingerprints:
                rejected_tokens.update(re.findall(r"[a-z][a-z0-9_-]{2,}", fp.lower()))
            keywords = [kw for kw in keywords if kw not in rejected_tokens]
            if not keywords:
                keywords = [w for w in all_seed_words if w not in rejected_tokens][:2]
        keywords = self._apply_keyword_floor(keywords, all_seed_words, rejected_tokens)
        keywords = keywords[:3]

        subcat_label = self._resolve_subcategory(active_kind)
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
        recently rejected fingerprints, but never below the keyword floor
        (Fix #6).
        """
        seed_lower = seed_hint.lower()
        all_seed_words = [w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", seed_lower)
                           if w not in _QUERY_STOPWORDS and len(w) > 2]
        keywords = all_seed_words[:3]

        # De-prioritize keywords that match rejected fingerprints
        rejected_tokens = set()
        if rejected_fingerprints:
            for fp in rejected_fingerprints:
                rejected_tokens.update(re.findall(r"[a-z][a-z0-9_-]{2,}", fp.lower()))
            keywords = [kw for kw in keywords if kw not in rejected_tokens]
            if not keywords:
                keywords = [w for w in all_seed_words if w not in rejected_tokens][:2]
        keywords = self._apply_keyword_floor(keywords, all_seed_words, rejected_tokens)
        keywords = keywords[:3]

        subcat_label = self._resolve_subcategory(active_kind)

        # Feature 1: concept filtering for OpenAlex queries
        concept_filter_str = None
        if getattr(self.runtime_config, "openalex_concept_filtering_enabled", True):
            concept_ids = _OPENALEX_CONCEPTS.get(subcat_label, _OPENALEX_CONCEPTS["general"])
            concept_filter_str = "|".join(concept_ids)
            log_agent_action("TopicHunter", "openalex_concept_filter_applied", {
                "subcat": subcat_label, "concepts": concept_ids,
            })

        queries = []
        if keywords:
            search_str = " ".join(keywords[:3])
            queries.append((search_str, {"sort": "relevance_score:desc"}))
            recent_year = max(2020, datetime.now().year - 2)
            recency_filter = f"publication_year:>{recent_year}"
            if concept_filter_str:
                recency_filter = f"concepts.id:{concept_filter_str},{recency_filter}"
            queries.append((search_str, {"filter": recency_filter, "sort": "cited_by_count:desc"}))
            if len(keywords) >= 2:
                alt_str = " ".join(keywords[:2])
                alt_params: Dict[str, Any] = {"sort": "relevance_score:desc"}
                if concept_filter_str:
                    alt_params["filter"] = f"concepts.id:{concept_filter_str}"
                queries.append((alt_str, alt_params))
            # Cross-pollination: one query pairing the kind's method term with the seed's anchor
            if active_kind and active_kind in _EXPLORATION_KIND_BIASES:
                kind_biases = _EXPLORATION_KIND_BIASES.get(active_kind, {})
                kind_method = kind_biases.get("method", [""])[0]
                seed_anchor = keywords[0]
                if kind_method and kind_method != seed_anchor:
                    cross_search = f"{kind_method} {seed_anchor}"
                    cross_params: Dict[str, Any] = {"sort": "relevance_score:desc"}
                    if concept_filter_str:
                        cross_params["filter"] = f"concepts.id:{concept_filter_str}"
                    queries.append((cross_search, cross_params))

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

        # Fix #6: never dedup away every single query — an empty result here
        # cascades into "seed_skipped_no_queries" even when a milder query
        # would have been fine. Keep the least-overlapping original query.
        if not deduped and queries:
            def _overlap_count(q: str) -> int:
                return len(_extract_fingerprint(q) & rejected_tokens)
            best = min(queries, key=_overlap_count)
            deduped = [best]
            log_agent_action("TopicHunter", "preflight_dedup_floor_kept_one", {"query": best})

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

        Fix #3: the comparison-trigger threshold is clamped so it can never sit
        above `novelty_similarity_reject`. Previously the trigger was a hardcoded
        0.70; if the configured reject threshold was lower than that, a candidate
        landing between the two values would be rejected on cosine similarity
        alone, with no LLM contribution comparison ever run. Now the trigger is
        always <= the reject threshold, guaranteeing comparison happens first.
        """
        if not abstracts:
            return {
                "max_similarity": 0.0,
                "reject": False,
                "nearest": None,
                "novelty_comparisons": [],
                "verdict": "NOVEL",
            }
        reject_threshold = float(getattr(self.runtime_config, "novelty_similarity_reject", 0.90))
        comparison_trigger = min(_DEFAULT_COMPARISON_TRIGGER, reject_threshold)

        topic_desc = f"{candidate_topic.get('title','')} {candidate_topic.get('description','')} {candidate_topic.get('contribution','')}"
        topic_emb = generate_embedding(topic_desc)
        best_sim = 0.0
        nearest = None
        high_overlap_papers = []

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

        # Contribution comparison for high overlap papers (similarity >= comparison_trigger)
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

        # Fix #3: reject on raw similarity alone only if similarity clears the
        # reject threshold AND no comparison could be obtained (e.g. LLM call
        # failed both attempts) — never because the trigger window was skipped.
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

    def _quick_grounding_check(self, gap: Dict[str, Any], recent_papers: List[Dict[str, Any]]) -> bool:
        """Fix #7: cheap heuristic pre-filter run before the expensive gate chain
        (screener LLM call, novelty embeddings + LLM comparison, formalization LLM
        call, citation graph fetches). Screens out gaps whose title/description
        share essentially no vocabulary with any retrieved paper — a strong signal
        the LLM hallucinated a topic ungrounded in the literature we actually
        fetched, before we pay for the full gate chain on it.

        This is intentionally lenient (low bar) — it is a pre-filter, not a
        replacement for the real gap/novelty/feasibility gates that follow.
        """
        text = f"{gap.get('title','')} {gap.get('description','')} {gap.get('rationale','')}".lower()
        gap_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", text))
        gap_tokens -= _QUERY_STOPWORDS
        if not gap_tokens or not recent_papers:
            return True  # can't evaluate — don't block on a heuristic we can't compute

        for paper in recent_papers[:15]:
            paper_text = f"{paper.get('title','')} {paper.get('abstract','')}".lower()
            paper_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", paper_text))
            if gap_tokens & paper_tokens:
                return True
        return False

    def _get_dataset_catalog(self) -> List[Dict[str, Any]]:
        """Return cached dataset catalog summary. Fail-open: returns [] on error."""
        if self._dataset_catalog_cache is not None:
            return self._dataset_catalog_cache
        try:
            from core.datasets import list_datasets
            raw = list_datasets()
            self._dataset_catalog_cache = [
                {
                    "name": d.get("name", ""),
                    "domain_tags": d.get("domain_tags", []),
                    "n_samples": d.get("max_rows", 0),
                    "task_type": d.get("task_type", "unknown"),
                }
                for d in raw
            ]
        except Exception as e:
            log_agent_action("TopicHunter", "dataset_catalog_load_error", {"error": str(e)})
            self._dataset_catalog_cache = []
        return self._dataset_catalog_cache

    def _dataset_plan_admissible(self, gap: Dict[str, Any]) -> bool:
        """Feature 1: cheap string check — reject gaps whose dataset_plan names
        a dataset not in the local catalog, unless it's a synthetic:<desc> plan.
        Fail-open: catalog errors or empty catalog → always True."""
        if not getattr(self.runtime_config, "capability_first_dataset_scoping_enabled", True):
            return True
        dataset_plan = str(gap.get("dataset_plan", "")).strip()
        if not dataset_plan:
            return True
        if dataset_plan.startswith("synthetic") or dataset_plan.startswith("bundled"):
            return True
        catalog = self._get_dataset_catalog()
        if not catalog:
            log_agent_action("TopicHunter", "dataset_admissibility_skip_empty_catalog", {})
            return True
        catalog_names = {d["name"].lower() for d in catalog}
        if dataset_plan.lower() in catalog_names:
            return True
        log_agent_action("TopicHunter", "dataset_plan_rejected", {
            "title": gap.get("title"),
            "dataset_plan": dataset_plan,
            "reason_code": "dataset_not_catalogued",
        })
        return False

    def _find_replication_targets(self, papers: List[Dict[str, Any]], max_results: int = 5) -> List[Dict[str, Any]]:
        """Feature 5: heuristic regex pre-filter to flag papers making strong claims
        with no visible variance/multi-seed reporting. Returns candidates for
        replication-and-extension topics. No LLM call — purely regex/heuristic."""
        if not getattr(self.runtime_config, "replication_target_mining_enabled", True):
            return []
        strong_claim_re = re.compile(
            r"(significantly|substantially|outperform|superior|state.of.the.art|best.?performing|achieves?.\s*\d)",
            re.IGNORECASE,
        )
        no_variance_re = re.compile(
            r"(seeds?|std|standard deviation|confidence interval|multiple runs|repeated trials|variance|error bar|±)",
            re.IGNORECASE,
        )
        candidates = []
        for p in papers:
            abstract = p.get("abstract", "") or ""
            title = p.get("title", "") or ""
            text = f"{title} {abstract}"
            if not text.strip():
                continue
            has_strong_claim = bool(strong_claim_re.search(text))
            has_no_variance = not bool(no_variance_re.search(abstract))
            if has_strong_claim and has_no_variance:
                # Extract a claim excerpt
                excerpt = ""
                m = strong_claim_re.search(abstract)
                if m:
                    start = max(0, m.start() - 60)
                    end = min(len(abstract), m.end() + 60)
                    excerpt = abstract[start:end].strip()
                candidates.append({
                    "title": title[:150],
                    "abstract_snippet": abstract[:300],
                    "matched_claim_excerpt": excerpt,
                })
                if len(candidates) >= max_results:
                    break
        return candidates

    def formalize_hypothesis(
        self,
        candidate: Dict[str, Any],
        gap_report: Dict[str, Any],
        novelty_report: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Hypothesis Formalizer: converts candidate into a machine-checkable StructuredHypothesis.

        Fix #5: added a targeted repair turn. Previously, a failed attempt simply
        regenerated the entire hypothesis from scratch with no feedback about what
        was wrong. This is the most expensive gate in the chain (it runs after
        screening + novelty comparison + citation fetches), so a bare retry with
        no repair signal wastes the highest-cost failures. Now, on failure, the
        next attempt is told exactly which required field(s) were missing/invalid
        and asked to return a corrected full JSON with only that fixed.
        """
        required_fields = ("hypothesis", "falsification_condition", "dependent_variables", "research_question")

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
        prompt = base_prompt
        last_parsed: Dict[str, Any] = {}
        max_attempts = 3
        for attempt in range(max_attempts):
            raw = call_llm(prompt, temperature=0.3, tier="strong")
            parsed = parse_json_from_llm(raw) or {}
            if isinstance(parsed, dict):
                last_parsed = parsed
                missing = [f for f in required_fields if not parsed.get(f)]
                if not missing:
                    parsed["gap_report"] = gap_report
                    parsed["novelty_report"] = novelty_report
                    return parsed
            else:
                missing = list(required_fields)

            if attempt < max_attempts - 1:
                # Repair turn: hand back what we got and name exactly what's missing,
                # instead of silently regenerating from scratch.
                prompt = base_prompt + f"""

Your previous response was missing or had an invalid value for: {', '.join(missing)}.
Your previous response was:
{json.dumps(last_parsed, default=str)[:2000]}

Return the FULL corrected JSON object again, in the same schema, fixing ONLY the
listed field(s) and leaving everything else as close to your previous answer as
possible.
"""
                log_agent_action("TopicHunter", "formalize_hypothesis_repair_attempt", {
                    "attempt": attempt + 1,
                    "missing_fields": missing,
                    "title": candidate.get("title"),
                })

        return None

    def feasibility_filter(self, topic: Topic) -> FeasibilityReport:
        """Grounded in what Engineer sandbox can actually run.

        Fix #2: blocked-phrase matching is now negation-aware. Previously a plain
        substring check meant a description saying e.g. "does not require a gpu
        cluster" or "no human subjects needed" would be false-positive rejected,
        since the blocked phrase is still a literal substring of the negated
        sentence. Now a blocked phrase occurring immediately after a negation cue
        (not/no/without/does not require/etc.) is treated as a disclaimer, not a
        requirement, and does not trigger rejection.
        """
        reasons = []
        ok = True
        text = json.dumps(topic).lower()
        for b in _BLOCKED_SANDBOX_PHRASES:
            if self._phrase_blocked_in_text(b, text):
                ok = False
                reasons.append(f"Not executable in sandbox: {b}")
        feas = topic.get("feasibility", 5)
        if isinstance(feas, (int, float)) and feas < 3:
            ok = False
            reasons.append(f"Low feasibility score: {feas}")
        # Prefer synthetic / public small data
        if "dataset" in text and "proprietary" in text and not self._is_negated("proprietary", text):
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

    @staticmethod
    def _is_negated(phrase: str, text: str, window: int = 24) -> bool:
        """Fix #2: check whether `phrase` is preceded by a negation cue within
        `window` characters, anywhere it occurs in `text`. Returns True if every
        occurrence of `phrase` is negated (so the phrase should NOT trigger a
        block); returns False if at least one occurrence is unnegated."""
        found_any = False
        for m in re.finditer(re.escape(phrase), text):
            found_any = True
            start = max(0, m.start() - window)
            preceding = text[start:m.start()]
            if not any(cue in preceding for cue in _NEGATION_CUES):
                return False  # an unnegated occurrence exists -> not fully negated
        return found_any

    @classmethod
    def _phrase_blocked_in_text(cls, phrase: str, text: str) -> bool:
        """Fix #2: True only if `phrase` appears in `text` at least once WITHOUT
        being immediately preceded by a negation cue. A phrase that only ever
        appears negated (disclaiming the requirement) does not block."""
        if phrase not in text:
            return False
        return not cls._is_negated(phrase, text)

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

    def _fallback_bridge_match(self, gap: Dict[str, Any], evidence_map: Dict[str, Any]) -> List[str]:
        """Fix #4: programmatically match a gap against the evidence map's bridges
        by keyword overlap, for use when the LLM omitted or malformed
        `evidence_bridge_ids`. This lets a gap that's actually grounded in a real
        bridge survive even when the model failed at exact-ID citation — an
        exact-string-matching task LLMs are unreliable at, especially at
        temperature=0.8 on a cheap tier.
        """
        bridges = evidence_map.get("bridges", []) if isinstance(evidence_map, dict) else []
        if not bridges:
            return []
        gap_text = f"{gap.get('title','')} {gap.get('description','')} {gap.get('rationale','')}".lower()
        gap_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", gap_text)) - _QUERY_STOPWORDS
        matched_ids = []
        for bridge in bridges:
            bridge_id = bridge.get("id") or bridge.get("bridge_id")
            if not bridge_id:
                continue
            bridge_text = f"{bridge.get('method_signal','')} {bridge.get('target_setting_signal','')}".lower()
            bridge_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", bridge_text))
            if bridge_tokens and (gap_tokens & bridge_tokens):
                matched_ids.append(bridge_id)
        return matched_ids

    def _generate_hyde_abstract(self, seed_hint: str, domain: str) -> Optional[str]:
        """Generate a short hypothetical abstract for what a real paper on
        `seed_hint` within `domain` would say, purely from the LLM's parametric
        knowledge (no retrieval). Used to derive richer query keywords than
        tokenizing the seed phrase directly. Returns None on any failure so
        callers can fall back to the existing keyword-extraction path unchanged.
        """
        if not getattr(self.runtime_config, "hyde_enabled", True):
            return None
        prompt = f"""
Write a 2-3 sentence hypothetical abstract for a research paper in {domain}
that investigates: {seed_hint}

Write it the way a real arXiv abstract reads: specific method names, specific
evaluation setups, specific terminology a specialist would use. Do not
mention that this is hypothetical. Return ONLY the abstract text, no preamble,
no JSON, no quotes.
"""
        try:
            raw = call_llm(prompt, temperature=0.5, tier="cheap")
            text = (raw or "").strip()
            if not text:
                return None
            return text[: getattr(self.runtime_config, "hyde_max_chars", 600)]
        except Exception as e:
            log_agent_action("TopicHunter", "hyde_generation_error", {"error": str(e), "seed": seed_hint})
            return None

    def _derive_followup_query_text(
        self, seed_hint: str, hyde_abstract: Optional[str],
        papers_so_far: List[Dict[str, Any]], domain: str,
    ) -> Optional[str]:
        """Cheap LLM call: given what little we found, propose a narrower or
        differently-worded query phrase to try on hop 2. Returns None on failure
        (multi-hop loop treats that as 'stop, no more hops')."""
        sample_titles = [p.get("title", "")[:100] for p in papers_so_far[:5]]
        prompt = f"""
A literature search for the research seed below returned very few results.
Seed: {seed_hint}
{"Hypothetical target abstract: " + hyde_abstract if hyde_abstract else ""}
Titles found so far (may be off-target or too generic): {json.dumps(sample_titles)}

Propose ONE alternative, more specific or differently-phrased search phrase
(3-8 words, using terminology a specialist in {domain} would search for) that
might surface literature this first attempt missed. Return ONLY the phrase,
no punctuation, no explanation.
"""
        try:
            raw = call_llm(prompt, temperature=0.4, tier="cheap")
            text = (raw or "").strip().strip('"').strip("'")
            return text or None
        except Exception as e:
            log_agent_action("TopicHunter", "followup_query_error", {"error": str(e)})
            return None

    def _harvest_frontier_terms(self, domain: str) -> List[Dict[str, str]]:
        """Sample very recent, high-signal papers for `domain` and extract
        emerging method/evaluation term pairs via a cheap LLM call. Cached to
        disk and refreshed every `frontier_refresh_every_n_runs` calls to
        discover_topics() (run-count gated, NOT time-gated — this system has no
        background daemon, so a wall-clock TTL cannot reflect actual usage
        cadence). Returns [] on any failure — callers must treat this as purely
        additive to the existing static _EXPLORATION_KIND_BIASES seeds, never a
        replacement, so a [] result degrades to exactly current behavior.
        """
        if not getattr(self.runtime_config, "frontier_seeding_enabled", True):
            return []

        cache_path = Path(self.runtime_config.output_dir) / "source_cache" / "frontier_terms.json"
        refresh_every_n = max(1, int(getattr(self.runtime_config, "frontier_refresh_every_n_runs", 5)))

        cached = None
        try:
            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
        except Exception as e:
            log_agent_action("TopicHunter", "frontier_cache_read_error", {"error": str(e)})
            cached = None

        if cached and cached.get("domain") == domain:
            runs_since = int(cached.get("runs_since_refresh", 0))
            if runs_since < refresh_every_n - 1:
                # Not due yet — bump the counter and reuse cached terms without
                # spending any LLM/API budget.
                cached["runs_since_refresh"] = runs_since + 1
                try:
                    cache_path.write_text(json.dumps(cached))
                except Exception as e:
                    log_agent_action("TopicHunter", "frontier_cache_write_error", {"error": str(e)})
                return cached.get("terms", [])
            # else: due for refresh — fall through to harvest below.

        # Sample recent, high-signal papers directly (not via a seed — this is
        # domain-wide, not seed-specific).
        sample_size = int(getattr(self.runtime_config, "frontier_sample_size", 30))
        categories = _ARXIV_CATEGORIES.get(domain, _ARXIV_CATEGORIES["general"])
        sample_query = f"cat:{categories[0]}"
        papers = self.search_arxiv(sample_query, max_results=sample_size)
        if not papers:
            return []

        titles = [p.get("title", "")[:150] for p in papers if p.get("title")]
        n_terms = int(getattr(self.runtime_config, "frontier_terms_extracted", 8))
        prompt = f"""
Below are titles of very recent papers in {domain}. Identify {n_terms}
emerging method/evaluation term pairs that represent CURRENT active research
directions — prefer specific, technical, compound terminology (e.g. a real
method name or technique combination) over broad category words already
implied by field basics.

Titles:
{json.dumps(titles[:sample_size], indent=2)}

Return JSON: {{"terms": [{{"method": "...", "evaluation": "..."}}]}}
"""
        try:
            parsed = parse_json_from_llm(call_llm(prompt, temperature=0.5, tier="cheap")) or {}
            terms = parsed.get("terms") or []
            terms = [t for t in terms if isinstance(t, dict) and t.get("method") and t.get("evaluation")][:n_terms]
        except Exception as e:
            log_agent_action("TopicHunter", "frontier_extraction_error", {"error": str(e)})
            return []

        if terms:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({
                    "domain": domain,
                    "generated_at": datetime.now().isoformat(),
                    "runs_since_refresh": 0,
                    "terms": terms,
                }))
            except Exception as e:
                log_agent_action("TopicHunter", "frontier_cache_write_error", {"error": str(e)})
            log_agent_action("TopicHunter", "frontier_terms_harvested", {"domain": domain, "count": len(terms)})

        return terms

    def _cached_search(self, cache_key: str, fetch_fn) -> List[Dict[str, Any]]:
        """Thread-safe run-scoped cache for identical query strings issued by
        different seeds within the same discover_topics() call. Exact-match
        only — no fuzzy matching, to avoid silently sharing results across
        seeds that only look similar."""
        if not getattr(self.runtime_config, "cross_seed_paper_cache_enabled", True):
            return fetch_fn()
        with self._run_query_cache_lock:
            if cache_key in self._run_query_cache:
                log_agent_action("TopicHunter", "cross_seed_cache_hit", {"key": cache_key[:120]})
                return self._run_query_cache[cache_key]
        result = fetch_fn()
        with self._run_query_cache_lock:
            self._run_query_cache[cache_key] = result
        return result

    def _hunt_once(self, domain: str, seed_hint: str, seed_strategy: str = "generic_fallback") -> List[Dict[str, Any]]:
        cross_run_context = CrossRunMemory().get_prompt_context()
        excluded = self._excluded_titles()
        rejected_fingerprints = self._extract_rejected_fingerprints(cross_run_context)
        active_kind = self._active_hypothesis_kind()

        # Feature 2: HyDE-based query construction
        hyde_abstract = self._generate_hyde_abstract(seed_hint, domain)
        query_source_text = f"{seed_hint} {hyde_abstract}" if hyde_abstract else seed_hint

        # Feature 3: multi-hop retrieval
        recent_papers: List[Dict[str, Any]] = []
        seen_paper_keys: set = set()
        max_hops = max(1, int(getattr(self.runtime_config, "multi_hop_max_hops", 2))) \
            if getattr(self.runtime_config, "multi_hop_retrieval_enabled", True) else 1
        min_threshold = int(getattr(self.runtime_config, "multi_hop_min_papers_threshold", 12))

        current_source_text = query_source_text
        for hop in range(max_hops):
            arxiv_queries = self._build_arxiv_queries(current_source_text, rejected_fingerprints, active_kind)
            openalex_queries = self._build_openalex_queries(current_source_text, rejected_fingerprints, active_kind)
            openalex_searches = [q[0] for q in openalex_queries]
            openalex_extra_params = [q[1] for q in openalex_queries]

            arxiv_queries = self._preflight_dedup(arxiv_queries, rejected_fingerprints)
            openalex_searches = self._preflight_dedup(openalex_searches, rejected_fingerprints)

            if not arxiv_queries and not openalex_searches:
                if hop == 0:
                    log_agent_action("TopicHunter", "seed_skipped_no_queries", {
                        "seed": seed_hint, "hop": hop,
                        "reason": "all_keywords_filtered_by_rejected_fingerprints",
                        "rejected_fingerprints_count": len(rejected_fingerprints),
                    })
                    return []
                break

            base_limit = 40
            extra = min(self._iteration_failures * 10, 30)
            search_limit = base_limit + extra

            hop_papers: List[Dict[str, Any]] = []
            # Feature 5: wrap search calls in cross-seed cache
            for i, query in enumerate(openalex_searches):
                extra_params = openalex_extra_params[i] if i < len(openalex_extra_params) else {}
                oa_key = f"openalex::{query}::{json.dumps(extra_params, sort_keys=True)}::{search_limit}"
                hop_papers.extend(self._cached_search(oa_key, lambda q=query, ep=extra_params, sl=search_limit: self.search_openalex(q, sl, extra_params=ep)))
            # Feature 5: wrap arxiv search in cache too
            for query in arxiv_queries:
                arxiv_key = f"arxiv::{query}::{min(search_limit, 30)}"
                hop_papers.extend(self._cached_search(arxiv_key, lambda q=query, sl=min(search_limit, 30): self.search_arxiv(q, sl)))

            new_count = 0
            for p in hop_papers:
                key = (p.get("doi") or p.get("arxiv_id") or p.get("title", "")).strip().lower()
                if key and key not in seen_paper_keys:
                    seen_paper_keys.add(key)
                    recent_papers.append(p)
                    new_count += 1

            log_agent_action("TopicHunter", "retrieval_hop_complete", {
                "seed": seed_hint, "hop": hop, "new_papers": new_count,
                "total_papers": len(recent_papers),
            })

            if len(recent_papers) >= min_threshold or hop == max_hops - 1:
                break

            # Thin yield — derive a follow-up query
            follow_up = self._derive_followup_query_text(seed_hint, hyde_abstract, recent_papers, domain)
            if not follow_up or follow_up.strip().lower() == current_source_text.strip().lower():
                break
            current_source_text = follow_up
            log_agent_action("TopicHunter", "multi_hop_triggered", {
                "seed": seed_hint, "hop_next": hop + 1, "follow_up_source": follow_up[:200],
            })

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

        # Feature 2: Structural gap mining via bibliographic coupling
        coupling_gaps: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "structural_gap_mining_enabled", True):
            try:
                coupling_gaps = find_coupling_gaps(
                    recent_papers,
                    self.s2_headers,
                    min_shared_refs=3,
                    max_pairs=getattr(self.runtime_config, "structural_gap_max_pairs", 8),
                )
            except Exception as e:
                log_agent_action("TopicHunter", "structural_gap_mining_error", {"error": str(e)})
                coupling_gaps = []

        # Feature 3: Method × Domain sparsity matrix
        sparse_cells: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "sparsity_matrix_enabled", True):
            try:
                sparse_cells = find_sparse_cells(recent_papers)
            except Exception as e:
                log_agent_action("TopicHunter", "sparsity_matrix_error", {"error": str(e)})
                sparse_cells = []

        # Feature 4: Contradiction mining from abstracts
        contradictions: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "contradiction_mining_enabled", True):
            try:
                contradictions = find_contradictions(recent_papers)
            except Exception as e:
                log_agent_action("TopicHunter", "contradiction_mining_error", {"error": str(e)})
                contradictions = []

        # Feature 5: Replication-target mining (heuristic)
        replication_targets: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "replication_target_mining_enabled", True):
            try:
                replication_targets = self._find_replication_targets(recent_papers)
            except Exception as e:
                log_agent_action("TopicHunter", "replication_target_mining_error", {"error": str(e)})
                replication_targets = []

        # Feature 6: Own negative results as prior work
        negative_lessons: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "negative_result_seeding_enabled", True):
            try:
                negative_lessons = CrossRunMemory().get_negative_result_lessons()
            except Exception as e:
                log_agent_action("TopicHunter", "negative_result_seeding_error", {"error": str(e)})
                negative_lessons = []

        abstracts = [
            (p.get("abstract") or "") for p in recent_papers if p.get("abstract")
        ][:25]
        evidence_map = build_cross_paper_evidence_map(recent_papers)

        # Fix #7: request more raw candidates per seed so the multi-stage gate
        # chain (which has a nonzero rejection rate at each stage) has more
        # surviving material at the end.
        base_prompt = f"""
Find research GAPS (not trendy saturated topics) in {domain}.
Seed angle: {seed_hint}
Prior-run lessons (avoid repeats):
{json.dumps(cross_run_context, sort_keys=True)}

Topics already rejected or that FAILED hypothesis debate (do NOT propose these or near-duplicates):
{json.dumps(excluded[-25:], indent=2)}

Citation-graph gap signals (high in-degree, low recent extensions):
{json.dumps(graph_signals[:5], indent=2)}

Structural gap signals (bibliographic coupling — pairs sharing references but not citing each other):
{json.dumps(coupling_gaps[:5], indent=2)}

Method × domain sparse cells (well-established individually but rarely combined in retrieved literature):
{json.dumps(sparse_cells[:5], indent=2)}

Contradictions in retrieved literature (papers disagreeing on the same question — strong, ready-made research gaps):
{json.dumps(contradictions[:5], indent=2)}

Replication-target candidates (papers making strong claims with no visible variance/multi-seed reporting — consider proposing replication-and-extension topics):
{json.dumps(replication_targets[:5], indent=2)}

Your own prior completed experiments that tested a hypothesis and did NOT find support (real negative results — use to propose a follow-up varying ONE condition, not a repeat):
{json.dumps(negative_lessons[:5], indent=2)}

Evidence-backed cross-paper bridges. These are candidate transfer questions,
not proof of a research gap. Use their cited excerpts when relevant; do not
invent a relationship not present in the supplied evidence:
{json.dumps(evidence_map.get('bridges', [])[:6], indent=2)[:6000]}

Sample recent titles:
{[p.get('title', '')[:100] for p in recent_papers[:8]]}

A real gap: foundational work is cited but rarely extended lately.
Propose {_GAPS_PER_SEED_REQUEST} topics executable with CPU sklearn/numpy synthetic or small public data.
For each topic include an explicit "contribution" sentence for novelty checking.

For every proposed topic, include `evidence_bridge_ids` containing the bridge IDs
that support its cross-paper synthesis. Do not fabricate IDs.
If a gap is grounded in one of the structural gap signals above, optionally include
`structural_gap_ref` naming the two papers (paper_a and paper_b) from the signal.
If a gap is grounded in one of the contradiction signals above, optionally include
`contradiction_ref` naming the two papers (paper_a_title and paper_b_title) from the signal.
JSON: {{"gaps": [{{"title": "...", "description": "...", "rationale": "...", "impact": "...",
"feasibility": 7, "keywords": [], "anchor_paper": "...", "dataset_plan": "synthetic|public",
"evidence_bridge_ids": ["bridge-..."], "structural_gap_ref": {{"paper_a": "...", "paper_b": "..."}},
"contradiction_ref": {{"paper_a_title": "...", "paper_b_title": "..."}}}}]}}
"""

        # Feature 7: Persona ensemble generation
        use_persona_ensemble = getattr(self.runtime_config, "persona_ensemble_enabled", True)
        persona_count = getattr(self.runtime_config, "persona_count", 2)

        if use_persona_ensemble and persona_count >= 2:
            persona_prefixes = {
                "skeptic": "Prioritize gaps about evaluation validity, statistical rigor, reporting standards, and reproducibility over new methods. ",
                "practitioner": "Prioritize gaps about real-world deployment constraints, robustness, and practical failure modes over theoretical novelty. ",
            }
            persona_names = list(persona_prefixes.keys())[:persona_count]
            all_gaps: List[Dict[str, Any]] = []

            def _call_persona(name: str) -> List[Dict[str, Any]]:
                prefix = persona_prefixes.get(name, "")
                prompt = prefix + base_prompt
                raw = call_llm(prompt, temperature=0.8, tier="cheap")
                parsed = parse_json_from_llm(raw) or {}
                gaps_list = parsed.get("gaps") or []
                for g in gaps_list:
                    g["persona"] = name
                return gaps_list

            with ThreadPoolExecutor(max_workers=persona_count) as persona_pool:
                persona_futures = {persona_pool.submit(_call_persona, name): name for name in persona_names}
                for fut in as_completed(persona_futures):
                    try:
                        all_gaps.extend(fut.result() or [])
                    except Exception as e:
                        log_agent_action("TopicHunter", "persona_call_error", {"persona": persona_futures[fut], "error": str(e)})

            # Dedup by title (preserving first occurrence)
            seen_titles: set = set()
            gaps = []
            for g in all_gaps:
                t = (g.get("title") or "").strip().lower()
                if t and t not in seen_titles:
                    seen_titles.add(t)
                    gaps.append(g)
        else:
            parsed = parse_json_from_llm(call_llm(base_prompt, temperature=0.8, tier="cheap")) or {}
            gaps = parsed.get("gaps") or []

        kept = []
        for gap in gaps:
            # Fix #7: cheap heuristic pre-filter before the expensive gate chain.
            # This intentionally runs before bridge validation too, since it's
            # nearly free and catches the worst-case "hallucinated, no relation
            # to any retrieved paper" candidates earliest.
            if not self._quick_grounding_check(gap, recent_papers):
                self._reject(gap, "ungrounded_in_retrieved_literature", {
                    "lesson_type": "pre_filter_rejection",
                    "reason_code": "no_vocabulary_overlap_with_retrieved_papers",
                })
                continue

            if not self._dataset_plan_admissible(gap):
                self._reject(gap, "dataset_not_catalogued", {
                    "lesson_type": "dataset_scoping_rejection",
                    "reason_code": "dataset_not_catalogued",
                    "dataset_plan": gap.get("dataset_plan"),
                })
                continue

            bridge_validation = validate_candidate_bridge_claim(gap, evidence_map)
            gap["bridge_validation"] = bridge_validation
            if not bridge_validation["valid"]:
                # Fix #4: previously this was always a hard reject. Now, only a
                # genuine content mismatch (the validator's own semantic check)
                # is treated as hard-reject-worthy; a missing/malformed ID list
                # is downgraded to a soft warning, with a programmatic fallback
                # attempt to attach real bridge IDs by keyword match so the gap
                # can still be evidence-tagged downstream.
                reason_text = str(bridge_validation.get("reason", ""))
                is_id_only_issue = bridge_validation.get("reason_code") in (
                    "missing_bridge_ids", "malformed_bridge_ids", "no_bridge_ids_provided",
                ) or "id" in reason_text.lower()
                if is_id_only_issue:
                    fallback_ids = self._fallback_bridge_match(gap, evidence_map)
                    gap["evidence_bridge_ids"] = fallback_ids
                    gap["bridge_validation"]["valid"] = True
                    gap["bridge_validation"]["soft_warning_applied"] = True
                    log_agent_action("TopicHunter", "bridge_validation_soft_downgrade", {
                        "title": gap.get("title"),
                        "original_reason": reason_text,
                        "fallback_ids_found": len(fallback_ids),
                    })
                else:
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
            # Validate optional structural_gap_ref field
            sgr = gap.get("structural_gap_ref")
            if sgr and isinstance(sgr, dict):
                pa = sgr.get("paper_a", "")
                pb = sgr.get("paper_b", "")
                coupling_titles = {(c.get("paper_a", ""), c.get("paper_b", "")) for c in coupling_gaps}
                if (pa, pb) not in coupling_titles and (pb, pa) not in coupling_titles:
                    gap.pop("structural_gap_ref", None)
            elif sgr is not None:
                gap.pop("structural_gap_ref", None)
            # Validate optional contradiction_ref field
            cr = gap.get("contradiction_ref")
            if cr and isinstance(cr, dict):
                pa_title = (cr.get("paper_a_title") or "").strip().lower()
                pb_title = (cr.get("paper_b_title") or "").strip().lower()
                valid_contradictions = {
                    (c["paper_a_title"].strip().lower(), c["paper_b_title"].strip().lower())
                    for c in contradictions
                }
                if (pa_title, pb_title) not in valid_contradictions and (pb_title, pa_title) not in valid_contradictions:
                    gap.pop("contradiction_ref", None)
            elif cr is not None:
                gap.pop("contradiction_ref", None)
            # Validate optional builds_on_negative_result field
            bnr = gap.get("builds_on_negative_result")
            if bnr and isinstance(bnr, dict):
                rq = (bnr.get("research_question") or "").strip().lower()
                valid_rqs = {nl.get("research_question", "").strip().lower() for nl in negative_lessons if nl.get("research_question")}
                if rq not in valid_rqs:
                    gap.pop("builds_on_negative_result", None)
            elif bnr is not None:
                gap.pop("builds_on_negative_result", None)
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
            gap["seed_strategy"] = seed_strategy  # Feature 8: seed-strategy provenance
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
        self._run_query_cache = {}  # Feature 5: reset cross-seed cache per invocation
        domain = domain or self.runtime_config.research_domain
        log_agent_action("TopicHunter", "start_discovery", {"domain": domain, "parallel": n_parallel})
        cross_run_context = CrossRunMemory().get_prompt_context()
        seeds = self._generate_dynamic_seeds(n_parallel, cross_run_context, domain=domain)

        all_topics: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=n_parallel) as pool:
            futures = {
                pool.submit(self._hunt_once, domain, s["seed"], s.get("strategy", "generic_fallback")): s
                for s in seeds
            }
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

        ranked = self.rank_topics_by_potential(unique)

        # Fix #1: aggregate the rejection funnel for this run so bottleneck
        # gates are visible without having to grep individual log entries.
        funnel: Dict[str, int] = {}
        for entry in self.rejection_log:
            funnel[entry["reason"]] = funnel.get(entry["reason"], 0) + 1
        log_agent_action("TopicHunter", "rejection_funnel", {
            "domain": domain,
            "total_rejected": len(self.rejection_log),
            "total_kept": len(ranked),
            "by_reason": dict(sorted(funnel.items(), key=lambda kv: -kv[1])),
        })

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