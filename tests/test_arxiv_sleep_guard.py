"""Test that the arXiv query loop does not contain a redundant time.sleep.

The old code had an explicit time.sleep(3.0) between arxiv_queries in the
main-loop hop (topic_hunter.py:1571, pre-fix). This is redundant with the
client's own delay_seconds=3.0 enforcement inside arxiv.Client, and must not
be re-introduced.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.topic_hunter import TopicHunterAgent


def _make_agent(tmp_path=None) -> TopicHunterAgent:
    agent = TopicHunterAgent.__new__(TopicHunterAgent)
    agent.context = None
    cfg = MagicMock()
    cfg.research_domain = "machine_learning"
    cfg.openalex_email = "test@example.com"
    cfg.semantic_scholar_api_key = ""
    cfg.novelty_similarity_reject = 0.92
    cfg.topic_exploration_every = 4
    cfg.topic_exploration_seed = 42
    cfg.output_dir = str(tmp_path or Path("/tmp/th_test"))
    cfg.openalex_concept_filtering_enabled = True
    cfg.hyde_enabled = False
    cfg.hyde_max_chars = 600
    cfg.multi_hop_retrieval_enabled = False
    cfg.multi_hop_min_papers_threshold = 12
    cfg.multi_hop_max_hops = 1
    cfg.frontier_seeding_enabled = False
    cfg.frontier_refresh_every_n_runs = 5
    cfg.frontier_sample_size = 30
    cfg.frontier_terms_extracted = 8
    cfg.persona_ensemble_enabled = False
    cfg.persona_count = 2
    cfg.replication_target_mining_enabled = False
    cfg.contradiction_mining_enabled = False
    cfg.sparsity_matrix_enabled = False
    cfg.structural_gap_mining_enabled = False
    cfg.structural_gap_max_pairs = 8
    cfg.negative_result_seeding_enabled = False
    cfg.seed_strategy_elo_enabled = False
    cfg.cross_seed_paper_cache_enabled = True
    cfg.capability_first_dataset_scoping_enabled = False
    agent.runtime_config = cfg
    agent._excluded_titles_cache = None
    agent._iteration_failures = 0
    agent._run_query_cache = {}
    agent._run_query_cache_lock = MagicMock()
    agent._dataset_catalog_cache = None
    agent.rejection_log = []
    agent.source_health = {}
    agent.source_client = MagicMock()
    agent.vector_memory = MagicMock()
    agent._rng = MagicMock()
    agent._arxiv_client = MagicMock()
    agent._arxiv_lock = MagicMock()
    return agent


class TestNoRedundantArxivSleep:
    """Verify time.sleep is not called when _build_arxiv_queries returns >1 query."""

    def test_no_sleep_between_arxiv_queries_in_hop_loop(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.capability_first_dataset_scoping_enabled = False
        agent.runtime_config.structural_gap_mining_enabled = False
        agent.runtime_config.sparsity_matrix_enabled = False
        agent.runtime_config.contradiction_mining_enabled = False
        agent.runtime_config.replication_target_mining_enabled = False
        agent.runtime_config.negative_result_seeding_enabled = False
        agent.runtime_config.persona_ensemble_enabled = False
        agent.runtime_config.seed_strategy_elo_enabled = False
        agent.runtime_config.hyde_enabled = False
        agent.runtime_config.multi_hop_retrieval_enabled = True
        agent.runtime_config.multi_hop_max_hops = 2
        agent.runtime_config.multi_hop_min_papers_threshold = 20
        agent.runtime_config.cross_seed_paper_cache_enabled = True

        fake_papers = [
            {"title": f"paper {i}", "abstract": f"abstract {i}",
             "doi": "", "arxiv_id": f"arxiv:{i}", "cited_by_count": 10}
            for i in range(25)
        ]

        formalized_hyp = {
            "hypothesis": "X improves Y",
            "falsification_condition": "No improvement",
            "dependent_variables": ["accuracy"],
            "research_question": "Does X improve Y?",
            "minimum_viable_experiment": {
                "dataset": "bundled_synthetic",
                "models": ["logistic_regression"],
                "conditions": ["control"],
                "metrics": ["accuracy"],
                "seeds": 3,
            },
        }

        with patch("agents.topic_hunter.CrossRunMemory") as MockCRM, \
             patch("agents.topic_hunter.call_llm") as mock_llm, \
             patch("agents.topic_hunter.parse_json_from_llm") as mock_parse, \
             patch("agents.topic_hunter.generate_embedding", return_value=[0.1]*10), \
             patch("agents.topic_hunter.build_cross_paper_evidence_map", return_value={"bridges": []}), \
             patch("agents.topic_hunter.validate_candidate_bridge_claim", return_value={"valid": True, "bridge_ids": []}), \
             patch("agents.topic_hunter.validate_topic_admission", return_value={"admitted": True, "errors": []}), \
             patch.object(agent, "_generate_hyde_abstract", return_value=None), \
             patch.object(agent, "_build_arxiv_queries", return_value=["query1", "query2", "query3"]), \
             patch.object(agent, "_build_openalex_queries", return_value=[("q", {})]), \
             patch.object(agent, "_preflight_dedup", side_effect=lambda q, _: q), \
             patch.object(agent, "search_openalex", return_value=fake_papers[:5]), \
             patch.object(agent, "search_arxiv", return_value=fake_papers[5:]), \
             patch.object(agent, "fetch_citation_graph", return_value=None), \
             patch.object(agent, "screen_research_gap", return_value={"status": "PASS", "gap_type": "supported"}), \
             patch.object(agent, "evaluate_layered_novelty", return_value={"reject": False, "verdict": "NOVEL", "max_similarity": 0.0}), \
             patch.object(agent, "feasibility_filter", return_value={"ok": True, "reasons": []}), \
             patch.object(agent, "formalize_hypothesis", return_value=formalized_hyp), \
             patch("agents.topic_hunter.time.sleep") as mock_sleep:

            crm_inst = MockCRM.return_value
            crm_inst.get_prompt_context.return_value = []
            crm_inst.excluded_topic_titles.return_value = []
            crm_inst.get_negative_result_lessons.return_value = []
            crm_inst.record_rejection.return_value = None

            mock_llm.return_value = json.dumps({"gaps": [{"title": "test gap", "description": "d", "rationale": "r", "contribution": "c", "evidence_bridge_ids": [], "feasibility": 7}]})
            mock_parse.return_value = {"gaps": [{"title": "test gap", "description": "d", "rationale": "r", "contribution": "c", "evidence_bridge_ids": [], "feasibility": 7}]}

            result = agent._hunt_once(domain="ml", seed_hint="test seed", seed_strategy="test")

            assert mock_sleep.call_count == 0, \
                f"time.sleep was called {mock_sleep.call_count} time(s) - redundant sleep reintroduced: {mock_sleep.call_args_list}"