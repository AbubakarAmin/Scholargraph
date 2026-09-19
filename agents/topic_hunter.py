"""
TopicHunterAgent — citation-graph gap analysis, novelty filter, parallel hunts.

CHANGELOG (this revision, v3):
  1. arXiv access moved to the official `arxiv` client (built-in 1 req / 3s
     pacing + 429/503 retry honoring Retry-After). The portalocker /tmp
     file-lock hack is gone; the API gateway (with its new adaptive AIMD
     bucket) paces and protects every provider call.
  2. API gateway v2: adaptive token buckets (rate cut on 429/503, additive
     recovery), backoff jitter, single-flight coalescing of identical reads,
     and is_available()/breaker_state() probes so open breakers short-circuit
     to a skip instead of paying retry sleeps.
  3. LLM-driven seed generation: one cheap call converts cross-run lessons +
     frontier terms into specific technical seed phrases (strategy
     "llm_diverse"), with graceful fallback to the static template seeds.
  4. Query building no longer turns seed filler words ("gaps", "problems",
     "open", "underexplored", ...) into literal search keywords, which used to
     retrieve noise literature and degrade gap quality downstream.
  5. Two-pool LLM budget: seed/query generation calls can no longer starve the
     gate chain — the gate chain reserves its own sub-budget, so gaps that
     were generated get evaluated instead of being silently discarded.
  6. literature_evidence is now relevance-ranked (embedding cosine with
     token-overlap fallback) instead of "first 8 papers fetched", so screener
     and novelty prompts judge gaps against the closest prior work.
  7. retrieve_literature(query) added — shared multi-source retrieval used by
     the QA-mode graph (previously referenced but missing).
  8. Structural gap mining now routes S2 calls through the API gateway
     (previously bypassed it with its own retry/sleep loop).
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

import numpy as np
import requests

from core.config import config
from core.utils import log_agent_action, parse_json_from_llm, calculate_similarity, title_token_overlap, call_llm_json
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


from core.structural_gaps import find_coupling_gaps
from core.sparsity_matrix import find_sparse_cells
from core.contradiction_mining import find_contradictions


logger = logging.getLogger(__name__)

# arXiv access policy (v3):
#   - The official `arxiv` pip client is used instead of hand-rolled HTTP. It
#     implements the documented 1 req / 3s pacing (delay_seconds), retries
#     429/503 with exponential backoff honoring Retry-After (num_retries), and
#     paginates at page_size. A module-level singleton client shares one
#     last-request clock across all TopicHunterAgent instances in this process.
#   - The gateway still wraps every search call (bucket pacing + circuit
#     breaker + health tracking) at one slot per search, and its adaptive
#     bucket learns from any 429/503 that leaks through.
#   - The previous cross-process portalocker file lock under /tmp was removed:
#     it duplicated gateway responsibilities and broke on Windows.
import arxiv as _arxiv_lib

_ARXIV_CLIENT: Optional["_arxiv_lib.Client"] = None
_ARXIV_CLIENT_LOCK = threading.Lock()


def _get_arxiv_client() -> "_arxiv_lib.Client":
    """Process-wide shared official arXiv client (shared delay clock)."""
    global _ARXIV_CLIENT
    if _ARXIV_CLIENT is None:
        with _ARXIV_CLIENT_LOCK:
            if _ARXIV_CLIENT is None:
                _ARXIV_CLIENT = _arxiv_lib.Client(
                    page_size=100,
                    delay_seconds=3.0,
                    num_retries=3,
                )
    return _ARXIV_CLIENT


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

# Known public ML datasets — if a gap's dataset_plan contains any of these
# substrings, it passes admissibility without checking the local catalog.
_KNOWN_PUBLIC_DATASETS = frozenset({
    "mnist", "cifar", "imagenet", "fashion", "svhn", "stl10",
    "glue", "superglue", "squad", "mnli", "mrpc", "qnli", "rte", "wnli", "cola", "stsb", "sst", "qqp",
    "commonsense", "piqa", "hellaswag", "winogrande", "arc",
    "openbookqa", "boolq", "commitmentbank", "swag",
    "imdb", "yelp", "ag news", "20newsgroups", "reuters",
    "iris", "wine", "breast cancer", "diabetes", "california housing",
    "boston housing", " Ames Housing",
    "librispeech", "common voice", "voxceleb",
    "coco", "pascal voc", "ade20k", "cityscapes", "gta5",
    "kitti", "waymo", "nuscenes",
    "omniglot", "miniimagenet", "tiered imagenet",
    "ptb", "wikitext", "text8",
    "mmlu", "humaneval", "mbpp", "gsm8k", "math",
    "pubmed", "arxiv", "reddit", "bookcorpus",
    "tabular", "csv", "uci",
})

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

# Seed phrases are meta-level descriptions ("open problems in attention",
# "methodological gaps in evaluation"). Searching arXiv/OpenAlex for the words
# "problems", "gaps", "open", "underexplored" retrieves noise, and noise
# literature poisons every downstream gate. These are excluded from keyword
# extraction before building provider queries. The keyword floor guarantees at
# least one content-bearing keyword still survives (see _apply_keyword_floor).
_SEED_FILLER_WORDS = frozenset({
    "gaps", "gap", "problems", "problem", "open", "challenges", "challenge",
    "underexplored", "unexplored", "understudied", "methodological",
    "methodology", "limitations", "limitation", "issues", "issue",
    "questions", "question", "areas", "area", "study", "studies", "research",
    "literature", "survey", "review", "future", "directions", "direction",
    "work", "works", "novel", "emerging", "critique", "critical", "insight",
    "insights", "pitfalls", "pitfall", "weakness", "weaknesses",
})


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
        self._iteration_failures = 0
        self._arxiv_consecutive_failures = 0
        self._arxiv_disabled = False
        # Feature 5: run-scoped, thread-safe query cache for cross-seed dedup
        self._run_query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._run_query_cache_lock = threading.Lock()
        self._dataset_catalog_cache: Optional[List[Dict[str, Any]]] = None

    def _source_ok(self, name: str):
        self.source_health[name] = {"ok": True}

    def _source_failed(self, name: str, error: Exception):
        self.source_health[name] = {"ok": False, "error": str(error)}

    def _s2_get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
        """Semantic Scholar GET routed through API gateway."""
        from core.api_gateway import get_gateway, RateLimitError

        gateway = get_gateway()
        try:
            if not gateway.is_available("s2"):
                return None
        except Exception:
            pass

        def _do_fetch():
            r = requests.get(
                url,
                headers=self.s2_headers,
                params=params or {},
                timeout=20,
            )
            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", "3"))
                raise RateLimitError("s2", retry_after)
            return r

        try:
            return gateway.request("s2", _do_fetch, retries=2, backoff_base=3.0)
        except Exception as e:
            logger.debug("S2 request failed for %s: %s", url, e)
            return None

    def search_openalex(self, query: str, limit: int = 50, extra_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        runtime_cfg = getattr(self, "runtime_config", None)
        if runtime_cfg is not None and not getattr(runtime_cfg, "openalex_enabled", True):
            log_agent_action("TopicHunter", "openalex_skipped_disabled", {"query": query[:80]})
            return []
        from core.api_gateway import get_gateway
        try:
            if not get_gateway().is_available("openalex"):
                log_agent_action("TopicHunter", "openalex_skipped_breaker_open", {"query": query[:80]})
                self._source_failed("openalex", RuntimeError("circuit open"))
                return []
        except Exception:
            pass
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
        """arXiv search via the official `arxiv` client, paced by the API gateway.

        The shared client enforces the documented 1 req / 3s gap and retries
        429/503 internally (honoring Retry-After). The gateway wraps the whole
        search as one call: token-bucket pacing, circuit breaking, health
        tracking, and adaptive rate learning. Supports an injected
        `_arxiv_client` for tests.
        """
        runtime_cfg = getattr(self, "runtime_config", None)
        if runtime_cfg is not None and not getattr(runtime_cfg, "arxiv_enabled", True):
            log_agent_action("TopicHunter", "arxiv_skipped_disabled", {"query": query[:80]})
            return []

        from core.api_gateway import get_gateway

        # Fast skip when the arXiv breaker is open — don't pay retry sleeps.
        try:
            if not get_gateway().is_available("arxiv"):
                log_agent_action("TopicHunter", "arxiv_skipped_breaker_open", {"query": query[:80]})
                self._source_failed("arxiv", RuntimeError("circuit open"))
                return []
        except Exception:
            pass

        def _do_search():
            search = _arxiv_lib.Search(
                query=query,
                max_results=min(max_results, 100),
                sort_by=_arxiv_lib.SortCriterion.SubmittedDate,
                sort_order=_arxiv_lib.SortOrder.Descending,
            )
            client = getattr(self, "_arxiv_client", None) or _get_arxiv_client()
            results = []
            for r in client.results(search):
                published = getattr(r, "published", None)
                year = 0
                try:
                    year = int(published.year) if published else 0
                except (AttributeError, ValueError, TypeError):
                    year = 0
                authors = []
                for a in (r.authors or []):
                    name = getattr(a, "name", None) or str(a)
                    if name:
                        authors.append(name)
                if r.title and r.title.strip():
                    results.append({
                        "title": r.title.strip().replace("\n", " "),
                        "abstract": (r.summary or "").strip().replace("\n", " "),
                        "year": year,
                        "authors": authors,
                        "arxiv_id": r.entry_id or "",
                        "categories": list(r.categories or []),
                    })
            return results

        try:
            results = get_gateway().request("arxiv", _do_search, retries=2, backoff_base=8.0, backoff_max=120.0)
            self._source_ok("arxiv")
            self._arxiv_consecutive_failures = 0
            return results or []
        except Exception as e:
            self._source_failed("arxiv", e)
            is_rate_limit = "rate limit" in str(e).lower() or "429" in str(e) or "503" in str(e)
            if is_rate_limit:
                self._arxiv_consecutive_failures += 1
            else:
                self._arxiv_consecutive_failures = 0
            log_level = logging.WARNING if is_rate_limit else logging.DEBUG
            logger.log(log_level, "arxiv search failed for query=%r: %s (rate_limit=%s)", query, e, is_rate_limit)
            log_agent_action("TopicHunter", "search_arxiv_error", {
                "error": str(e), "query": query, "rate_limit": is_rate_limit,
            })
            return []

    def search_openalex_multi(self, queries: List[Tuple[str, Dict[str, Any]]], limit: int = 30) -> List[Dict[str, Any]]:
        """Run multiple OpenAlex queries with optional extra params and merge results."""
        all_results = []
        for query, params in queries:
            all_results.extend(self.search_openalex(query, limit, extra_params=params))
        return all_results

    def retrieve_literature(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """Multi-source literature retrieval for QA mode (v3 fix #7).

        Previously referenced by `workflow_nodes.qa_literature_retrieval_node`
        but never implemented — QA runs crashed with AttributeError. Queries
        OpenAlex + arXiv + S2 bulk in one pass, dedupes by DOI/arXiv-id/title,
        and returns {"query", "papers", "sources_used"}.

        Raises ResearchSourceUnavailable only when NO papers were found AND
        every attempted source is unhealthy; partial results degrade
        gracefully like discover_topics().
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("retrieve_literature requires a non-empty query")

        papers: List[Dict[str, Any]] = []
        seen: set = set()

        def _add(rows: List[Dict[str, Any]], source: str) -> int:
            added = 0
            for p in rows or []:
                if not p.get("title"):
                    continue
                key = (p.get("doi") or p.get("arxiv_id") or p.get("title", "")).strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                p.setdefault("source", source)
                papers.append(p)
                added += 1
            return added

        sources_used: Dict[str, int] = {}
        try:
            sources_used["openalex"] = _add(self.search_openalex(query, limit=limit), "openalex")
        except Exception as e:
            log_agent_action("TopicHunter", "retrieve_literature_openalex_error", {"error": str(e)})
            sources_used["openalex"] = 0
        try:
            sources_used["arxiv"] = _add(self.search_arxiv(query, max_results=limit), "arxiv")
        except Exception as e:
            log_agent_action("TopicHunter", "retrieve_literature_arxiv_error", {"error": str(e)})
            sources_used["arxiv"] = 0
        try:
            from core.sources_s2_bulk import search_s2_bulk
            s2_rows = search_s2_bulk(query, self.s2_headers, limit=limit)
            sources_used["s2_bulk"] = _add(s2_rows, "s2_bulk")
        except Exception as e:
            log_agent_action("TopicHunter", "retrieve_literature_s2_error", {"error": str(e)})
            sources_used["s2_bulk"] = 0

        if not papers:
            any_healthy = any(s.get("ok") for s in self.source_health.values())
            if not any_healthy and sources_used:
                details = "; ".join(f"{n}: {e.get('error', 'unavailable')}" for n, e in self.source_health.items())
                raise ResearchSourceUnavailable(
                    "QA literature retrieval could not contact any scholarly source. "
                    "Check network access and OPENALEX_EMAIL, then try again. Details: " + details
                )

        log_agent_action("TopicHunter", "retrieve_literature_complete", {
            "query": query[:120], "papers": len(papers), "sources_used": sources_used,
        })
        return {"query": query, "papers": papers, "sources_used": sources_used}

    def search_arxiv_multi(self, queries: List[str], max_results: int = 30) -> List[Dict[str, Any]]:
        """Run multiple arXiv queries and merge results.

        Pacing is delegated entirely to the shared arXiv client (3s gap) and
        the gateway bucket — no extra inter-query sleep layer. Identical
        duplicate queries are collapsed to a single network call.
        """
        all_results: List[Dict[str, Any]] = []
        seen_queries: set = set()
        for query in queries:
            key = query.strip().lower()
            if not key or key in seen_queries:
                continue
            seen_queries.add(key)
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

    def _generate_llm_seeds(
        self,
        domain: str,
        cross_run_context: List[Dict[str, Any]],
        negative_lessons: Optional[List[Dict[str, Any]]] = None,
        n_seeds: int = 6,
    ) -> List[Dict[str, str]]:
        """LLM-driven seed generation (v3 fix #3).

        Static template seeds ("... open problems", "... underexplored
        challenges") recur every run and produce generic queries. This asks the
        LLM — once, cheap tier — to mint specific, technical seed phrases from
        the structured signals we already have (cross-run rejection tags,
        negative results). Returns [] on any failure; callers fall back to the
        static seed banks unchanged.
        """
        if not getattr(self.runtime_config, "llm_seed_generation_enabled", True):
            return []
        runtime_cfg = getattr(self, "runtime_config", None)
        if not domain:
            domain = getattr(runtime_cfg, "research_domain", None) if runtime_cfg is not None else None
        if not domain:
            return []
        prompt = f"""
You are designing literature-search seeds for an automated research system in {domain}.
Each seed is a short phrase (4-10 words) naming a SPECIFIC technical research angle —
name concrete methods, model families, or measurement setups, not meta-vocabulary.
Bad: "open problems in evaluation". Good: "speculative decoding verification overhead".

Prior-run context (avoid re-proposing rejected/failed areas):
{json.dumps(cross_run_context[-10:], sort_keys=True, default=str)[:1500]}

Return JSON with {n_seeds} DIVERSE seeds spread across different subfields
and method families, each with a short strategy tag explaining its angle:
{{"seeds": [{{"seed": "...", "angle": "one-line rationale"}}]}}
"""
        try:
            raw = call_llm(prompt, temperature=0.9, tier="cheap")
            parsed = parse_json_from_llm(raw) or {}
            items = parsed.get("seeds") or parsed.get("topics") or []
            seeds: List[Dict[str, str]] = []
            for item in items:
                if isinstance(item, dict) and item.get("seed"):
                    text = str(item["seed"]).strip()
                    if 8 <= len(text) <= 120:
                        seeds.append({"seed": text, "strategy": "llm_diverse"})
                elif isinstance(item, str) and 8 <= len(item.strip()) <= 120:
                    seeds.append({"seed": item.strip(), "strategy": "llm_diverse"})
            if seeds:
                log_agent_action("TopicHunter", "llm_seeds_generated", {"count": len(seeds)})
            return seeds[:n_seeds]
        except Exception as e:
            log_agent_action("TopicHunter", "llm_seed_generation_error", {"error": str(e)})
            return []

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

        # 0. v3 fix #3: LLM-minted seeds first (specific, technical, run-aware).
        # Empty on failure/disabled — everything below degrades to the static
        # template seeds exactly as before.
        negative_lessons: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "negative_result_seeding_enabled", True):
            try:
                negative_lessons = CrossRunMemory().get_negative_result_lessons()
            except Exception:
                negative_lessons = []
        llm_seeds = self._generate_llm_seeds(
            domain or self.runtime_config.research_domain,
            cross_run_context,
            negative_lessons=negative_lessons,
            n_seeds=max(3, n_seeds // 2),
        )
        for d in llm_seeds:
            words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", d["seed"].lower()))
            if not (words & rejected_tokens):
                seeds.append(d)

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
                    all_strategies = ["llm_diverse", "kind_bias", "cross_pollination", "generic_fallback", "frontier"]
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
        the keyword floor (Fix #6). Seed filler words ("gaps", "problems",
        "open", ...) are never used as search keywords (v3 fix #4).
        """
        seed_lower = seed_hint.lower()
        all_seed_words = [w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", seed_lower)
                           if w not in _QUERY_STOPWORDS and w not in _SEED_FILLER_WORDS and len(w) > 2]
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

        return queries[:2]

    def _build_openalex_queries(
        self,
        seed_hint: str,
        rejected_fingerprints: List[str],
        active_kind: Optional[str] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Build 2-4 OpenAlex structured queries with filter params.

        hypothesis kinds when one is active.  Excludes keywords matching
        recently rejected fingerprints, but never below the keyword floor
        (Fix #6). Seed filler words ("gaps", "problems", "open", ...) are
        never used as search keywords (v3 fix #4).
        """
        seed_lower = seed_hint.lower()
        all_seed_words = [w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", seed_lower)
                           if w not in _QUERY_STOPWORDS and w not in _SEED_FILLER_WORDS and len(w) > 2]
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
            r = self._s2_get(url, params={"fields": fields})
            if r is None or r.status_code != 200:
                return {}
            paper = r.json()
            # Recent citing papers (proxy for out-degree from recent work extending it)
            cites_url = f"{self.base_urls['s2']}/paper/{paper_id}/citations"
            c = self._s2_get(
                cites_url,
                params={"fields": "citingPaper.year,citingPaper.title", "limit": 50},
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

    def evaluate_layered_novelty(
        self,
        candidate_topic: Dict[str, Any],
        abstracts: List[Dict[str, Any]],
        precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Layered novelty evaluation:
        1. Fast embedding similarity check (uses precomputed embeddings when available)
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

        for paper in abstracts[:20]:
            abs_text = paper.get("abstract") if isinstance(paper, dict) else str(paper)
            if not abs_text:
                continue
            # Use precomputed embedding if available, otherwise compute on the fly
            paper_key = (paper.get("doi") or paper.get("arxiv_id") or paper.get("title", "")).strip().lower() if isinstance(paper, dict) else None
            if precomputed_embeddings and paper_key and paper_key in precomputed_embeddings:
                emb = precomputed_embeddings[paper_key]
            else:
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
        parsed = call_llm_json(prompt, temperature=0.3, tier="cheap", attempts=2, call_fn=call_llm)
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

    def _rank_papers_for_gap(
        self,
        gap: Dict[str, Any],
        papers: List[Dict[str, Any]],
        top_n: int = 8,
    ) -> List[Dict[str, Any]]:
        """Rank retrieved papers by relevance to a candidate gap (v3 fix #6).

        Previously `literature_evidence` was simply the first 8 fetched papers,
        so the screener/novelty prompts often judged gaps against off-topic
        text. Scores here are lexical Jaccard overlap (deterministic, free);
        ties preserve retrieval order. The closest prior work — not arbitrary
        recent papers — is what novelty screening should compare against.
        """
        gap_text = " ".join(str(gap.get(k, "")) for k in ("title", "description", "contribution", "rationale")).lower()
        gap_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", gap_text)) - _QUERY_STOPWORDS - _SEED_FILLER_WORDS
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, p in enumerate(papers):
            if not (p.get("title") and p.get("abstract")):
                continue
            paper_text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
            paper_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", paper_text)) - _QUERY_STOPWORDS
            inter = gap_tokens & paper_tokens
            union = gap_tokens | paper_tokens
            score = len(inter) / (len(union) or 1)
            scored.append((score, -idx, p))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [p for _, _, p in scored[:top_n]]

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
        """Feature 1: string check — reject gaps whose dataset_plan names
        a dataset not in the local catalog, unless it's synthetic/bundled/known-public.
        Fail-open: catalog errors or empty catalog → always True."""
        if not getattr(self.runtime_config, "capability_first_dataset_scoping_enabled", True):
            return True
        dataset_plan = str(gap.get("dataset_plan", "")).strip()
        if not dataset_plan:
            return True
        if dataset_plan.startswith("synthetic") or dataset_plan.startswith("bundled"):
            return True
        # Accept common public ML benchmarks by name pattern
        plan_lower = dataset_plan.lower()
        for kw in _KNOWN_PUBLIC_DATASETS:
            if kw in plan_lower:
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
        """Hypothesis Formalizer: 2-phase approach for high reliability.

        Phase 1 (cheap tier): Generate ONLY the 4 core fields — research_question,
        hypothesis, falsification_condition, dependent_variables. If this fails,
        reject early without burning strong-tier calls.

        Phase 2 (strong tier): Build the FULL experimental contract around the
        validated core. This is a much easier task — the model just needs to
        flesh out an existing hypothesis, not invent one from scratch.
        """
        # --- Phase 1: core fields (cheap tier, fast, high pass rate) ---
        phase1_prompt = f"""
Given this research gap, produce EXACTLY 4 fields. Be concrete and specific.

Title: {candidate.get('title')}
Description: {candidate.get('description')}
Rationale: {candidate.get('rationale')}
Literature evidence: {json.dumps(novelty_report, default=str)[:800]}

Generate:
1. research_question: A precise, specific question (not a vague direction)
2. hypothesis: A falsifiable claim in the form "X causes/improves Y over Z because W"
3. falsification_condition: What specific observation would DISPROVE this hypothesis?
4. dependent_variables: Specific measurable metrics (e.g. ["ECE", "accuracy", "F1"])

Return JSON:
{{
  "research_question": "...",
  "hypothesis": "...",
  "falsification_condition": "...",
  "dependent_variables": ["..."]
}}
"""
        core_fields = ("research_question", "hypothesis", "falsification_condition", "dependent_variables")
        core_parsed: Dict[str, Any] = {}

        for attempt in range(2):
            temp = 0.3 if attempt == 0 else 0.1
            raw = call_llm(phase1_prompt, temperature=temp, tier="cheap")
            parsed = parse_json_from_llm(raw) or {}
            if isinstance(parsed, dict):
                missing = [f for f in core_fields if not parsed.get(f)]
                if not missing:
                    core_parsed = parsed
                    break
                if attempt == 0:
                    phase1_prompt += f"\n\nYour previous response was missing: {', '.join(missing)}. Fix ONLY those fields."

        if not core_parsed:
            log_agent_action("TopicHunter", "formalize_phase1_failed", {
                "title": candidate.get("title"),
                "missing": [f for f in core_fields if not core_parsed.get(f)],
            })
            return None

        log_agent_action("TopicHunter", "formalize_phase1_ok", {
            "title": candidate.get("title"),
            "core_fields": list(core_parsed.keys()),
        })

        # --- Phase 2: full contract (strong tier, easier task) ---
        required_fields = ("hypothesis", "falsification_condition", "dependent_variables", "research_question")

        base_prompt = f"""
Build a complete experimental contract around this validated hypothesis core.

VALIDATED CORE (do NOT change these — they are confirmed):
{json.dumps(core_parsed, indent=2)}

Candidate context:
Title: {candidate.get('title')}
Description: {candidate.get('description')}
Gap Report: {json.dumps(gap_report, default=str)[:1500]}
Sandbox manifest: {json.dumps(SANDBOX_CAPABILITY_MANIFEST.as_dict())}

Now ADD the following fields to complete the contract:
1. independent_variables: test ranges/values (e.g. [{{"name": "...", "values": [...]}}])
2. expected_relationship: prediction under control conditions
3. novelty_claim: what is new vs prior work
4. closest_prior_work: [{{
    "title": "...",
    "difference": "..."
  }}]
5. baselines: specific named baselines (e.g. ["logistic_regression", "random_forest"])
6. metrics: same as dependent_variables or additional
7. confounders: factors that could confound results
8. competing_explanations: alternative explanations the experiment must rule out
9. minimum_viable_experiment: {{
    "dataset": "bundled_synthetic" or specific public dataset name,
    "models": ["specific_model_a", "specific_model_b"],
    "conditions": ["controlled_seeds"],
    "metrics": ["..."],
    "seeds": 3,
    "baseline": "specific_baseline",
    "expected_result": "...",
    "falsification_test": "Welch t-test p<0.05"
  }}
10. required_resources: {{"cpu": true, "gpu": false, "max_memory_mb": 4096, "max_runtime_seconds": 120}}

Return the COMPLETE JSON with ALL fields (core + new):
{{
  "research_question": "{core_parsed.get('research_question', '')}",
  "hypothesis": "{core_parsed.get('hypothesis', '')}",
  "falsification_condition": "{core_parsed.get('falsification_condition', '')}",
  "dependent_variables": {json.dumps(core_parsed.get('dependent_variables', []))},
  "independent_variables": [{{"name": "...", "values": [...]}}],
  "expected_relationship": "...",
  "novelty_claim": "...",
  "closest_prior_work": [{{"title": "...", "difference": "..."}}],
  "baselines": ["..."],
  "metrics": ["..."],
  "confounders": ["..."],
  "competing_explanations": ["..."],
  "minimum_viable_experiment": {{
    "dataset": "bundled_synthetic",
    "models": ["..."],
    "conditions": ["controlled_seeds"],
    "metrics": ["..."],
    "seeds": 3,
    "baseline": "...",
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
        max_attempts = 2
        for attempt in range(max_attempts):
            temp = 0.1 if attempt > 0 else 0.3
            raw = call_llm(prompt, temperature=temp, tier="strong")
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
                prompt = base_prompt + f"""

Your previous response was missing or had an invalid value for: {', '.join(missing)}.
Your previous response:
{json.dumps(last_parsed, default=str)[:2000]}

Return the FULL corrected JSON, fixing ONLY the listed field(s).
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

    def pre_debate_self_critique(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        """Reflection agent: identifies top 3 likely debate objections and rewrites
        the hypothesis to preempt them. Burns 1 cheap-tier LLM call but
        significantly strengthens the hypothesis before adversarial debate.

        Returns the topic dict with strengthened structured_hypothesis,
        or the original topic if self-critique fails.
        """
        structured_hyp = topic.get("structured_hypothesis")
        if not structured_hyp or not isinstance(structured_hyp, dict):
            return topic

        prompt = f"""
You are a hostile peer reviewer. Given this hypothesis, identify the top 3
objections a reviewer would raise, then REWRITE the hypothesis to preempt them.

Hypothesis: {structured_hyp.get('hypothesis', '')}
Research question: {structured_hyp.get('research_question', '')}
Falsification condition: {structured_hyp.get('falsification_condition', '')}
Dependent variables: {json.dumps(structured_hyp.get('dependent_variables', []))}
Baselines: {json.dumps(structured_hyp.get('baselines', []))}
MVE: {json.dumps(structured_hyp.get('minimum_viable_experiment', {}), default=str)[:1000]}

For each objection:
1. What is the objection? (be specific)
2. How severe is it? (1-5)
3. How can the hypothesis be REWRITTEN to address it?

Then provide a STRENGTHENED version of the hypothesis that preempts all 3 objections
while preserving the core scientific claim.

Return JSON:
{{
  "objections": [
    {{"objection": "...", "severity": 3, "fix": "..."}}
  ],
  "strengthened_hypothesis": "...",
  "strengthened_falsification": "...",
  "strengthened_baselines": ["..."],
  "strengthened_mve": {{
    "dataset": "...",
    "models": ["..."],
    "baseline": "...",
    "metrics": ["..."],
    "seeds": 3,
    "falsification_test": "..."
  }}
}}
"""
        try:
            raw = call_llm(prompt, temperature=0.3, tier="cheap")
            parsed = parse_json_from_llm(raw) or {}
            if not isinstance(parsed, dict):
                return topic

            strengthened_hyp = parsed.get("strengthened_hypothesis")
            if not strengthened_hyp:
                return topic

            # Apply strengthening to the structured hypothesis
            new_hyp = dict(structured_hyp)
            new_hyp["hypothesis"] = strengthened_hyp

            strengthened_fals = parsed.get("strengthened_falsification")
            if strengthened_fals:
                new_hyp["falsification_condition"] = strengthened_fals

            strengthened_baselines = parsed.get("strengthened_baselines")
            if isinstance(strengthened_baselines, list) and strengthened_baselines:
                new_hyp["baselines"] = strengthened_baselines

            strengthened_mve = parsed.get("strengthened_mve")
            if isinstance(strengthened_mve, dict) and strengthened_mve:
                old_mve = new_hyp.get("minimum_viable_experiment") or {}
                old_mve.update({k: v for k, v in strengthened_mve.items() if v})
                new_hyp["minimum_viable_experiment"] = old_mve

            topic["structured_hypothesis"] = new_hyp
            topic["falsifiable_prediction"] = new_hyp.get("falsification_condition", "")

            objections = parsed.get("objections") or []
            log_agent_action("TopicHunter", "pre_debate_critique", {
                "title": topic.get("title"),
                "objections_found": len(objections),
                "hypothesis_strengthened": True,
            })
            return topic

        except Exception as e:
            log_agent_action("TopicHunter", "pre_debate_critique_error", {"error": str(e)})
            return topic

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

    def _suppress_near_duplicates(self, topics: List[Dict[str, Any]], threshold: float = 0.70) -> List[Dict[str, Any]]:
        """Drop later topics whose title overlaps an already-kept topic (v3).

        Uses the same token-overlap metric as failed-topic matching. Keeps the
        first occurrence (higher-ranked seed strategies were processed first).
        Falls back to all topics on any error — this is an optimization, not a
        gate.
        """
        try:
            kept: List[Dict[str, Any]] = []
            for t in topics:
                title = (t.get("title") or "").strip()
                if not title:
                    continue
                duplicate = any(
                    title_token_overlap(title, k.get("title", "")) >= threshold
                    for k in kept
                )
                if duplicate:
                    log_agent_action("TopicHunter", "near_duplicate_suppressed", {
                        "title": title[:120],
                        "against": kept[-1].get("title", "") if kept else "",
                        "threshold": threshold,
                    })
                    continue
                kept.append(t)
            return kept
        except Exception as e:
            log_agent_action("TopicHunter", "near_duplicate_suppression_error", {"error": str(e)})
            return list(topics)

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
        # domain-wide, not seed-specific). Falls back to OpenAlex if arXiv disabled.
        sample_size = int(getattr(self.runtime_config, "frontier_sample_size", 30))
        if getattr(self.runtime_config, "arxiv_enabled", True):
            categories = _ARXIV_CATEGORIES.get(domain, _ARXIV_CATEGORIES["general"])
            sample_query = f"cat:{categories[0]}"
            papers = self.search_arxiv(sample_query, max_results=sample_size)
        else:
            # Fallback: use OpenAlex with concept filtering for recent papers
            papers = self.search_openalex(domain, limit=sample_size)
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
            # S2 bulk search — free, higher throughput than arXiv, reduces arXiv dependency
            from core.sources_s2_bulk import search_s2_bulk
            s2_key = f"s2_bulk::{current_source_text}::{search_limit}"
            hop_papers.extend(self._cached_search(
                s2_key,
                lambda q=current_source_text, sl=search_limit: search_s2_bulk(q, self.s2_headers, limit=sl),
            ))

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

        # OpenReview reviewer-weakness signals — pre-formalized, expert-stated gaps,
        # not LLM-inferred from abstracts. Cached per discover_topics() run via
        # _run_query_cache so all 5+ seeds share one fetch, not 5.
        openreview_signals: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "openreview_enabled", True):
            try:
                from core.sources_openreview import get_openreview_client
                or_client = get_openreview_client()
                venue_keys = getattr(self.runtime_config, "openreview_venues", ["iclr2025", "neurips2024"])
                or_key = f"openreview::{','.join(sorted(venue_keys))}"
                openreview_signals = self._cached_search(or_key, lambda: [
                    sig for vk in venue_keys
                    for sig in or_client.fetch_venue_gap_signals(vk, max_submissions=40)
                ])
            except Exception as e:
                log_agent_action("TopicHunter", "openreview_fetch_error", {"error": str(e)})
                openreview_signals = []

        for p in openreview_signals:
            key = (p.get("title") or "").strip().lower()
            if key and key not in seen_paper_keys:
                seen_paper_keys.add(key)
                recent_papers.append(p)

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

        # Phase 3.1: Precompute abstract embeddings once for all gaps
        novelty_max_abstracts = int(getattr(self.runtime_config, "novelty_max_abstracts", 20))
        abstract_embeddings: Dict[str, np.ndarray] = {}
        for p in recent_papers[:novelty_max_abstracts]:
            abs_text = (p.get("abstract") or "")[:2000]
            if not abs_text:
                continue
            paper_key = (p.get("doi") or p.get("arxiv_id") or p.get("title", "")).strip().lower()
            if paper_key:
                abstract_embeddings[paper_key] = generate_embedding(abs_text)

        # Phase 3.2: LLM call budget per seed — TWO POOLS (v3 fix #5).
        # Previously one shared counter covered generation (personas) AND the
        # gate chain, so heavy generation could exhaust the budget and silently
        # discard every generated gap before evaluation (the "found but never
        # judged" failure mode). Generation now draws from a small dedicated
        # pool; the gate chain gets a protected reserve of the main budget.
        try:
            raw_budget = getattr(self.runtime_config, "llm_budget_per_seed", 30)
            llm_budget = max(4, int(raw_budget)) if isinstance(raw_budget, (int, float)) and not isinstance(raw_budget, bool) else 30
        except (TypeError, ValueError):
            llm_budget = 30
        gate_reserve = min(max(6, llm_budget // 4), 12)
        _llm_calls_used = {"count": 0, "generation": 0, "gate": 0}

        def _check_llm_budget(kind: str = "gate") -> bool:
            if kind == "generation":
                # Generation must leave room for the gate chain to run.
                if _llm_calls_used["count"] + gate_reserve >= llm_budget:
                    return False
            elif _llm_calls_used["count"] >= llm_budget:
                return False
            _llm_calls_used["count"] += 1
            if kind == "generation":
                _llm_calls_used["generation"] += 1
            return True

        # Fix #7: request more raw candidates per seed so the multi-stage gate
        # chain (which has a nonzero rejection rate at each stage) has more
        # surviving material at the end.
        base_prompt = f"""
Propose {_GAPS_PER_SEED_REQUEST} TESTABLE HYPOTHESES in {domain}.

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

Replication-target candidates (papers making strong claims with no visible variance/multi-seed reporting):
{json.dumps(replication_targets[:5], indent=2)}

Reviewer-identified weaknesses from peer review (OpenReview) — EXPERT-STATED limitations:
{json.dumps([{
    'title': p['title'],
    'weaknesses': p['weaknesses'][:2],
} for p in openreview_signals[:6]], indent=2)[:3000]}

Prior negative results (propose a follow-up varying ONE condition, not a repeat):
{json.dumps(negative_lessons[:5], indent=2)}

Evidence-backed cross-paper bridges:
{json.dumps(evidence_map.get('bridges', [])[:6], indent=2)[:6000]}

Sample recent titles:
{[p.get('title', '')[:100] for p in recent_papers[:8]]}

IMPORTANT: Each hypothesis MUST be a specific, falsifiable claim. NOT a vague research direction.

Each hypothesis should follow this form:
"Method/technique X applied to problem Y achieves Z improvement over baseline W because of mechanism M"

Requirements for each hypothesis:
- Named method (contrastive learning, dropout, ensemble, temperature scaling, specific architecture — NOT "a new approach")
- Named baseline to compare against (logistic regression, random forest, ResNet-18, temperature scaling — NOT "existing methods")
- Named metric (accuracy, ECE, F1, inference latency in ms — NOT "performance")
- Named dataset OR "synthetic with N samples, K features, D classes"
- Concrete prediction (e.g., "improves ECE by >15% over temperature scaling")
- Why this hasn't been done before (grounded in the evidence above)

BAD examples (will be rejected downstream):
- "Evaluating the impact of domain shift on calibration" ← too vague, not falsifiable
- "A study of interpretability methods under distribution shift" ← survey, not hypothesis
- "Characterizing robustness boundaries" ← not a testable claim

GOOD examples:
- "Temperature scaling outperforms Platt scaling on CIFAR-10-C when corruption severity > 3, because temperature parameters capture output distribution shifts better than sigmoid transforms"
- "Mixup training reduces ECE by >20% over standard training on synthetic data with Gaussian noise, because it smooths the decision boundary"
- "Random forests achieve higher accuracy than XGBoost on tabular data with >30% missing values, because tree-based methods handle missingness natively without imputation"

For every proposed hypothesis, include `evidence_bridge_ids` containing bridge IDs
that support its cross-paper synthesis. Do not fabricate IDs.
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

            for name in persona_names:
                if not _check_llm_budget("generation"):
                    log_agent_action("TopicHunter", "llm_budget_exhausted", {"seed": seed_hint, "at": "persona", "used": _llm_calls_used["count"]})
                    break
                try:
                    prefix = persona_prefixes.get(name, "")
                    prompt = prefix + base_prompt
                    raw = call_llm(prompt, temperature=0.8, tier="cheap")
                    parsed = parse_json_from_llm(raw) or {}
                    gaps_list = parsed.get("gaps") or []
                    for g in gaps_list:
                        g["persona"] = name
                    all_gaps.extend(gaps_list)
                except Exception as e:
                    log_agent_action("TopicHunter", "persona_call_error", {"persona": name, "error": str(e)})

            # Dedup by title (preserving first occurrence)
            seen_titles: set = set()
            gaps = []
            for g in all_gaps:
                t = (g.get("title") or "").strip().lower()
                if t and t not in seen_titles:
                    seen_titles.add(t)
                    gaps.append(g)
        else:
            if _check_llm_budget("generation"):
                parsed = parse_json_from_llm(call_llm(base_prompt, temperature=0.8, tier="cheap")) or {}
                gaps = parsed.get("gaps") or []
            else:
                log_agent_action("TopicHunter", "llm_budget_exhausted", {"seed": seed_hint, "at": "generation", "used": _llm_calls_used["count"]})
                gaps = []

        kept = []
        for gap in gaps:
            # Budget guard: skip expensive gate chain if LLM budget exhausted
            if not _check_llm_budget():
                log_agent_action("TopicHunter", "llm_budget_exhausted", {"seed": seed_hint, "at": "gate_chain", "used": _llm_calls_used["count"], "remaining_gaps": len(gaps) - len(kept)})
                break
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
            # v3 fix #6: evidence = the MOST RELEVANT retrieved papers, not the
            # first 8 fetched. Screener + novelty prompts now judge each gap
            # against its closest prior work.
            evidence_papers = self._rank_papers_for_gap(gap, recent_papers, top_n=8)
            gap["literature_evidence"] = [
                {
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", "")[:3000],
                    "doi": paper.get("doi"),
                    "arxiv_id": paper.get("arxiv_id"),
                }
                for paper in evidence_papers
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
            novelty_eval = self.evaluate_layered_novelty(gap, gap.get("literature_evidence", []), precomputed_embeddings=abstract_embeddings)
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
        for seed in seeds:
            try:
                topics = self._hunt_once(domain, seed["seed"], seed.get("strategy", "generic_fallback"))
                all_topics.extend(topics or [])
            except Exception as e:
                log_agent_action("TopicHunter", "hunt_error", {"seed": seed["seed"], "error": str(e)})

        # Graceful degradation: if we got partial results from at least one source, return them
        if not all_topics:
            if any(s.get("ok") for s in self.source_health.values()):
                log_agent_action("TopicHunter", "partial_results_returned", {
                    "source_health": self.source_health,
                })
            else:
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
        # v3: near-duplicate suppression across seeds — different seeds can
        # surface the same idea in different wording; exact-title dedup misses
        # that and debate slots get burned on redundant candidates.
        unique = self._suppress_near_duplicates(unique)

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

    def _execute_second_pass(self, top_bridges: List[Dict[str, Any]], domain: str, precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None) -> List[Topic]:
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

        if getattr(self.runtime_config, "arxiv_enabled", True):
            second_pass_papers = self.search_arxiv_multi(second_pass_queries[:4], 20)
        else:
            # Fallback: run the same queries through OpenAlex
            second_pass_papers = []
            for q in second_pass_queries[:4]:
                second_pass_papers.extend(self.search_openalex(q, limit=10))
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
            novelty_eval = self.evaluate_layered_novelty(gap, gap.get("literature_evidence", []), precomputed_embeddings=precomputed_embeddings)
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