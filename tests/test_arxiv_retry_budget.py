"""Test that search_arxiv does not multiply retry attempts.

The arxiv library (v4.0.1) has its own internal retry cascade (num_retries
attempts). Our search_arxiv has an outer retry loop (max_attempts=4). If both
layers retry independently, a single logical query can issue up to 16 real HTTP
requests (4 outer x 4 inner). This test file verifies the fix: num_retries=0
on the Client, with our outer loop owning all retry logic exclusively.

Two scenarios are tested:
1. Sustained 429s: total real HTTP request count stays bounded at max_attempts.
2. Transient errors (ConnectionError, UnexpectedEmptyPageError): still retried
   gracefully by our outer loop, not crashing _hunt_once.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import threading

import arxiv
import pytest
import requests

from agents.topic_hunter import TopicHunterAgent


def _make_agent(tmp_path=None) -> TopicHunterAgent:
    """Construct a minimal TopicHunterAgent with mocked dependencies."""
    agent = TopicHunterAgent.__new__(TopicHunterAgent)
    agent.context = None
    cfg = MagicMock()
    cfg.research_domain = "machine_learning"
    cfg.openalex_email = "test@example.com"
    cfg.semantic_scholar_api_key = ""
    cfg.novelty_similarity_reject = 0.92
    cfg.topic_exploration_every = 4
    cfg.topic_exploration_seed = 42
    cfg.output_dir = str(tmp_path or "/tmp/th_test")
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
    agent._run_query_cache_lock = threading.Lock()
    agent._dataset_catalog_cache = None
    agent.rejection_log = []
    agent.source_health = {}
    agent.source_client = MagicMock()
    agent.vector_memory = MagicMock()
    agent._rng = MagicMock()
    # Use a real Client with num_retries=0 so the test exercises our outer loop
    agent._arxiv_client = arxiv.Client(page_size=100, delay_seconds=0, num_retries=0)
    agent._arxiv_lock = threading.Lock()
    return agent


class TestBoundedRetryUnder429:
    """Sustained 429s must not produce more than max_attempts real HTTP requests."""

    def test_total_requests_bounded_under_sustained_429(self, tmp_path):
        agent = _make_agent(tmp_path)
        request_count = 0
        lock = threading.Lock()

        def fake_results(search):
            """Simulate arxiv.Client.results: yield after making a real HTTP request."""
            nonlocal request_count
            with lock:
                request_count += 1
            # Always raise HTTPError(429) — the library won't retry since num_retries=0
            raise arxiv.HTTPError(
                url="https://export.arxiv.org/api/query?test=1",
                retry=0,
                status=429,
            )

        agent._arxiv_client.results = fake_results

        with patch("agents.topic_hunter.time.sleep"):
            result = agent.search_arxiv("cat:cs.LG", max_results=10)

        assert result == [], "Should return empty list after exhausting retries"
        # max_attempts=4, so at most 4 real HTTP requests
        assert request_count <= 4, (
            f"Expected at most 4 real HTTP requests (max_attempts=4), "
            f"got {request_count}. Retry multiplication is still happening."
        )

    def test_exact_request_count_matches_max_attempts(self, tmp_path):
        """Every outer-loop attempt triggers exactly one real HTTP request."""
        agent = _make_agent(tmp_path)
        request_count = 0

        def fake_results(search):
            nonlocal request_count
            request_count += 1
            raise arxiv.HTTPError(url="test", retry=0, status=429)

        agent._arxiv_client.results = fake_results

        with patch("agents.topic_hunter.time.sleep"):
            agent.search_arxiv("test query", max_results=10)

        assert request_count == 4, (
            f"Expected exactly 4 real HTTP requests (max_attempts=4), "
            f"got {request_count}"
        )

    def test_library_num_retries_is_zero(self, tmp_path):
        """Confirm the Client is configured with num_retries=0."""
        agent = _make_agent(tmp_path)
        assert agent._arxiv_client.num_retries == 0, (
            f"Expected num_retries=0, got {agent._arxiv_client.num_retries}"
        )


class TestTransientErrorResilience:
    """Transient errors (ConnectionError, UnexpectedEmptyPageError) should be
    retried by our outer loop and not crash _hunt_once."""

    def test_connection_error_retried_then_succeeds(self, tmp_path):
        agent = _make_agent(tmp_path)
        call_count = 0

        def fake_results(search):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First two calls: transient network failure
                raise requests.exceptions.ConnectionError("Connection refused")
            # Third call: succeeds
            fake_result = MagicMock()
            fake_result.title = "Test Paper"
            fake_result.summary = "Test abstract"
            fake_result.published = MagicMock()
            fake_result.published.year = 2025
            fake_result.authors = [MagicMock(name="Author A")]
            fake_result.entry_id = "http://arxiv.org/abs/2501.00001"
            fake_result.categories = ["cs.LG"]
            return iter([fake_result])

        agent._arxiv_client.results = fake_results

        with patch("agents.topic_hunter.time.sleep"):
            result = agent.search_arxiv("test query", max_results=10)

        assert len(result) == 1, "Should succeed after retrying transient errors"
        assert result[0]["title"] == "Test Paper"
        assert call_count == 3, f"Expected 3 calls (2 failures + 1 success), got {call_count}"

    def test_unexpected_empty_page_retried_then_succeeds(self, tmp_path):
        agent = _make_agent(tmp_path)
        call_count = 0

        def fake_results(search):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: empty page (transient arXiv glitch)
                raise arxiv.UnexpectedEmptyPageError(
                    url="test", retry=0,
                    raw_feed=arxiv._feed.ParsedFeed(
                        header=arxiv._feed.FeedHeader(),
                        results=[],
                    ),
                )
            # Second call: succeeds
            fake_result = MagicMock()
            fake_result.title = "Test Paper"
            fake_result.summary = "Test abstract"
            fake_result.published = MagicMock()
            fake_result.published.year = 2025
            fake_result.authors = [MagicMock(name="Author A")]
            fake_result.entry_id = "http://arxiv.org/abs/2501.00001"
            fake_result.categories = ["cs.LG"]
            return iter([fake_result])

        agent._arxiv_client.results = fake_results

        with patch("agents.topic_hunter.time.sleep"):
            result = agent.search_arxiv("test query", max_results=10)

        assert len(result) == 1, "Should succeed after retrying empty page"
        assert call_count == 2, f"Expected 2 calls (1 failure + 1 success), got {call_count}"

    def test_permanent_failure_returns_empty_not_crash(self, tmp_path):
        """Non-retryable error (e.g. 400 Bad Request) returns [] without crashing."""
        agent = _make_agent(tmp_path)

        def fake_results(search):
            raise arxiv.HTTPError(url="test", retry=0, status=400)

        agent._arxiv_client.results = fake_results

        with patch("agents.topic_hunter.time.sleep"):
            result = agent.search_arxiv("test query", max_results=10)

        assert result == [], "Non-retryable error should return empty list"

    def test_all_retries_exhausted_returns_empty(self, tmp_path):
        """When all max_attempts are exhausted, returns [] gracefully."""
        agent = _make_agent(tmp_path)
        call_count = 0

        def fake_results(search):
            nonlocal call_count
            call_count += 1
            raise requests.exceptions.ConnectionError("Persistent failure")

        agent._arxiv_client.results = fake_results

        with patch("agents.topic_hunter.time.sleep"):
            result = agent.search_arxiv("test query", max_results=10)

        assert result == [], "Should return empty after all retries exhausted"
        assert call_count == 4, f"Expected exactly 4 calls (max_attempts), got {call_count}"
