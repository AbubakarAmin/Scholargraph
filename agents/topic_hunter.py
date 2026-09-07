"""
TopicHunterAgent — citation-graph gap analysis, novelty filter, parallel hunts.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


_FAILED_TOPIC_OVERLAP_THRESHOLD = 0.55


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

    def _source_ok(self, name: str):
        self.source_health[name] = {"ok": True}

    def _source_failed(self, name: str, error: Exception):
        self.source_health[name] = {"ok": False, "error": str(error)}

    def search_openalex(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
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
            artifact = self.source_client.fetch_json(
                "openalex",
                url,
                headers=self.openalex_headers,
                params=params,
                validator=lambda payload: isinstance(payload, dict) and "results" in payload,
            )
            if artifact["status"] == "unavailable":
                raise RuntimeError("; ".join(artifact.get("warnings", [])) or "source unavailable")
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

    def novelty_score(self, topic_desc: str, abstracts: List[str]) -> Dict[str, Any]:
        """High similarity to recent abstracts → reject (already published)."""
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
        for b in blocked:
            if b in text:
                ok = False
                reasons.append(f"Not executable in sandbox: {b}")
        feas = topic.get("feasibility", 5)
        if isinstance(feas, (int, float)) and feas < 4:
            ok = False
            reasons.append(f"Low feasibility score: {feas}")
        # Prefer synthetic / public small data
        if "dataset" in text and "proprietary" in text:
            ok = False
            reasons.append("Proprietary dataset unavailable")
        candidate_plan = {"methodology": topic.get("description", ""), "experiments": [{"dataset": {"name": topic.get("dataset_plan", "")}}]}
        capability_reasons = check_plan_feasibility(candidate_plan, SANDBOX_CAPABILITY_MANIFEST)
        if capability_reasons:
            ok = False
            reasons.extend(capability_reasons)
        return {"ok": ok, "reasons": reasons}

    def _reject(self, topic: Topic, reason: str, meta: Optional[Dict[str, Any]] = None):
        entry = {
            "title": topic.get("title"),
            "reason": reason,
            "meta": meta or {},
            "ts": datetime.now().isoformat(),
        }
        self.rejection_log.append(entry)
        CrossRunMemory().record_rejection("topic", topic.get("title", "?"), reason, meta)
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
        lessons = json.dumps(CrossRunMemory().get_prompt_context(), sort_keys=True)
        excluded = self._excluded_titles()
        recent_papers = self.search_openalex(f"{domain} {seed_hint}", 40)
        recent_papers.extend(self.search_arxiv(f"{domain} {seed_hint}", 20))
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

        prompt = f"""
Find research GAPS (not trendy saturated topics) in {domain}.
Seed angle: {seed_hint}
Prior-run lessons (avoid repeats):
{lessons}

Topics already rejected or that FAILED hypothesis debate (do NOT propose these or near-duplicates):
{json.dumps(excluded[-25:], indent=2)}

Citation-graph gap signals (high in-degree, low recent extensions):
{json.dumps(graph_signals[:5], indent=2)}

Sample recent titles:
{[p.get('title', '')[:100] for p in recent_papers[:8]]}

A real gap: foundational work is cited but rarely extended lately.
Propose 3 topics executable with CPU sklearn/numpy synthetic or small public data.
For each topic include an explicit "contribution" sentence for novelty checking.

JSON: {{"gaps": [{{"title": "...", "description": "...", "rationale": "...", "impact": "...",
"feasibility": 7, "keywords": [], "anchor_paper": "...", "dataset_plan": "synthetic|public"}}]}}
"""
        parsed = parse_json_from_llm(call_llm(prompt, temperature=0.8, tier="cheap")) or {}
        gaps = parsed.get("gaps") or []

        kept = []
        for gap in gaps:
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
            prior = self._matches_excluded_topic(gap.get("title", ""))
            if prior:
                self._reject(gap, "previously_failed_or_rejected", {"matched": prior})
                continue
            # Novelty
            nov = self.novelty_score(
                f"{gap.get('title','')} {gap.get('description','')} {gap.get('contribution','')}",
                abstracts,
            )
            gap["novelty"] = nov
            if nov.get("reject"):
                self._reject(gap, "novelty_too_low", nov)
                continue
            feas = self.feasibility_filter(gap)
            gap["feasibility_check"] = feas
            if not feas["ok"]:
                self._reject(gap, "infeasible_for_engineer", feas)
                continue
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

    def discover_topics(self, domain: str = None, n_parallel: int = 3) -> List[Topic]:
        self._excluded_titles_cache = None
        domain = domain or self.runtime_config.research_domain
        log_agent_action("TopicHunter", "start_discovery", {"domain": domain, "parallel": n_parallel})
        seeds = [
            "underexplored methods",
            "evaluation methodology gaps",
            "robustness and reproducibility",
        ][:n_parallel]

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

        # Deduplicate by title
        seen = set()
        unique = []
        for t in all_topics:
            title = (t.get("title") or "").lower().strip()
            if title and title not in seen:
                seen.add(title)
                unique.append(t)

        ranked = self.rank_topics_by_potential(unique)
        log_agent_action("TopicHunter", "discovery_complete", {
            "num_topics": len(ranked),
            "rejected": len(self.rejection_log),
        })
        return ranked

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
