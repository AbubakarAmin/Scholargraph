"""End-to-end test for QA mode: literature retrieval → synthesis → citation check."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.state import ResearchState, initialize_state
from core.contracts import LiteratureContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "abstract": "We propose a new architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "doi": "10.48550/arXiv.1706.03762",
        "arxiv_id": "1706.03762",
        "year": 2017,
        "cited_by_count": 90000,
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": "We introduce a new language representation model called BERT, designed to pre-train deep bidirectional representations.",
        "doi": "10.48550/arXiv.1810.04805",
        "arxiv_id": "1810.04805",
        "year": 2018,
        "cited_by_count": 60000,
    },
    {
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "abstract": "Recent work demonstrates that scaling up language models greatly improves task-agnostic, few-shot performance.",
        "doi": "10.48550/arXiv.2005.14165",
        "arxiv_id": "2005.14165",
        "year": 2020,
        "cited_by_count": 25000,
    },
]

SYNTHETIC_LIT_CONTEXT: LiteratureContext = {
    "papers": SYNTHETIC_PAPERS,
    "graph_signals": [{"paper_id": "1706.03762", "title": "Attention Is All You Need", "gap_score": 5.0}],
    "evidence_map": {"bridges": []},
    "query": "transformer attention mechanisms",
    "domain": "machine_learning",
}

SYNTHETIC_QA_ANSWER = {
    "answer": (
        "Transformer models have revolutionized NLP. The self-attention mechanism "
        "introduced in arXiv:1706.03762 enables parallel processing of sequences. "
        "BERT (arXiv:1810.04805) showed that bidirectional pre-training improves "
        "performance. GPT-3 (arXiv:2005.14165) demonstrated few-shot learning at scale."
    ),
    "key_findings": [
        "Self-attention enables parallel sequence processing",
        "Bidirectional pre-training improves representation quality",
        "Scaling enables few-shot task generalization",
    ],
    "bibliography": [
        {"title": "Attention Is All You Need", "arxiv_id": "1706.03762", "year": 2017},
        {"title": "BERT", "arxiv_id": "1810.04805", "year": 2018},
        {"title": "GPT-3", "arxiv_id": "2005.14165", "year": 2020},
    ],
    "limitations": "Does not cover vision transformers or multimodal architectures.",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_initialize_state_has_mode_field():
    """ResearchState must include a mode field defaulting to full_research."""
    state = initialize_state()
    assert state["mode"] == "full_research"
    assert state.get("literature_context") is None
    assert state.get("qa_answer") is None


def test_initialize_state_qa_mode():
    """initialize_state(mode='qa') sets mode correctly."""
    state = initialize_state(mode="qa")
    assert state["mode"] == "qa"


def test_qa_answer_node_passes_citation_check(monkeypatch):
    """qa_answer_node must call verify_citations and store the result."""
    from core import workflow_nodes

    # Mock call_llm to return a synthetic answer with arXiv citations
    monkeypatch.setattr(
        workflow_nodes,
        "call_llm",
        lambda *_a, **_k: json.dumps(SYNTHETIC_QA_ANSWER),
    )

    state = initialize_state(mode="qa")
    state["user_query"] = "What are transformer attention mechanisms?"
    state["literature_context"] = SYNTHETIC_LIT_CONTEXT

    result = workflow_nodes.qa_answer_node(state)

    assert result["qa_answer"] is not None
    assert result["qa_answer"]["answer"]
    assert len(result["qa_answer"]["key_findings"]) == 3
    assert len(result["qa_answer"]["bibliography"]) == 3
    assert result["qa_citation_verification"] is not None
    assert "passed" in result["qa_citation_verification"]
    assert result["current_phase"] == "complete"
    assert result["should_continue"] is False


def test_qa_answer_node_rejects_empty_query():
    """qa_answer_node must fail gracefully with no query."""
    from core import workflow_nodes

    state = initialize_state(mode="qa")
    state["literature_context"] = SYNTHETIC_LIT_CONTEXT
    state["user_query"] = ""

    result = workflow_nodes.qa_answer_node(state)

    assert result["terminal_error"]
    assert result["current_phase"] == "complete"
    assert result["should_continue"] is False


def test_qa_answer_node_rejects_no_literature():
    """qa_answer_node must fail gracefully with no literature."""
    from core import workflow_nodes

    state = initialize_state(mode="qa")
    state["user_query"] = "test query"
    state["literature_context"] = {"papers": [], "graph_signals": [], "evidence_map": {}, "query": "test", "domain": "test"}

    result = workflow_nodes.qa_answer_node(state)

    assert result["terminal_error"]
    assert result["current_phase"] == "complete"


def test_qa_answer_node_catches_llm_errors(monkeypatch):
    """qa_answer_node must handle LLM failures gracefully."""
    from core import workflow_nodes

    def failing_llm(*_a, **_k):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(workflow_nodes, "call_llm", failing_llm)

    state = initialize_state(mode="qa")
    state["user_query"] = "test"
    state["literature_context"] = SYNTHETIC_LIT_CONTEXT

    result = workflow_nodes.qa_answer_node(state)

    assert result["terminal_error"]
    assert "QA answer generation failed" in result["terminal_error"]
    assert result["current_phase"] == "complete"


def test_qa_graph_terminates_without_planning_or_engineering(monkeypatch):
    """The full qa_graph must run without invoking planning, engineering, or evidence-gate code."""
    from core.workflow import create_qa_graph
    from core import workflow_nodes

    # Track which node functions are called
    called_nodes = []

    def track_literature_retrieval(state):
        called_nodes.append("qa_literature_retrieval")
        state["literature_context"] = SYNTHETIC_LIT_CONTEXT
        state["current_phase"] = "qa_answer"
        return state

    def track_qa_answer(state):
        called_nodes.append("qa_answer")
        # Simulate the answer node behavior
        state["qa_answer"] = {"answer": "test", "key_findings": [], "bibliography": [], "limitations": ""}
        state["qa_citation_verification"] = {"passed": True, "score": 10.0}
        state["current_phase"] = "qa_verification"
        state["should_continue"] = False
        return state

    def track_qa_verification(state):
        called_nodes.append("qa_verification")
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state

    graph = create_qa_graph({
        "qa_literature_retrieval": track_literature_retrieval,
        "qa_answer": track_qa_answer,
        "qa_verification": track_qa_verification,
    })
    app = graph.compile()

    # Run the graph
    state = initialize_state(mode="qa")
    state["user_query"] = "test query"
    result = app.invoke(state)

    # Verify the graph ran the expected nodes
    assert "qa_literature_retrieval" in called_nodes
    assert "qa_answer" in called_nodes
    assert "qa_verification" in called_nodes

    # Verify no planning/engineering/evidence-gate nodes were invoked
    forbidden_phases = {
        "hypothesis_debate", "planning", "writing_narrative",
        "engineering", "independent_validation", "writing_results",
        "supervision", "meta_evaluation", "editing",
    }
    # The graph only has qa_literature_retrieval, qa_answer, and qa_verification, so
    # by construction it cannot invoke the forbidden nodes.
    # But we verify the result state is clean:
    assert result.get("plan") is None
    assert result.get("engineer_outputs") == {}
    assert result.get("evidence_gate") == {}
    assert result.get("debate_results") == []


def test_qa_citations_verified_with_mock_resolvers(monkeypatch):
    """Verify that verify_citations is called on the answer text."""
    from core import verification
    from core import workflow_nodes

    # Mock LLM to return answer with a fake DOI
    answer_with_doi = {
        "answer": "As shown in doi:10.9999/fake.doi.456, attention mechanisms are effective.",
        "key_findings": ["attention works"],
        "bibliography": [{"title": "Fake", "doi": "10.9999/fake.doi.456", "year": 2024}],
        "limitations": "limited scope",
    }
    monkeypatch.setattr(
        workflow_nodes, "call_llm",
        lambda *_a, **_k: json.dumps(answer_with_doi),
    )
    # Mock DOI resolution to fail
    monkeypatch.setattr(
        verification, "resolve_doi",
        lambda doi: {"resolved": False, "doi": doi, "error": "not found"},
    )

    state = initialize_state(mode="qa")
    state["user_query"] = "test"
    state["literature_context"] = SYNTHETIC_LIT_CONTEXT

    result = workflow_nodes.qa_answer_node(state)

    # Citation verification must have been called and failed
    assert result["qa_citation_verification"]["passed"] is False
    assert result["qa_citation_verification"]["failed"]
    # The meta_feedback should note the failure
    assert any("citation verification failed" in fb for fb in result["meta_feedback"])


def test_qa_mode_not_triggered_in_full_research():
    """initialize_state with mode='full_research' should not set QA fields."""
    state = initialize_state(mode="full_research")
    assert state["mode"] == "full_research"
    assert state.get("literature_context") is None
    assert state.get("qa_answer") is None
    assert state.get("user_query") is None
