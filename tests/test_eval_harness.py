"""
Eval harness — measures whether upgrades help (hard checks, sandbox, falsifiability).
Run: pytest tests/ -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. Sandbox lockdown
# ---------------------------------------------------------------------------

def test_sandbox_blocks_subprocess():
    from core.sandbox import validate_code, execute_sandboxed

    ok, err = validate_code("import subprocess\nsubprocess.run(['echo','hi'])")
    assert not ok
    assert "blocked" in err.lower() or "Import" in err


def test_sandbox_blocks_exit():
    from core.sandbox import validate_code

    ok, err = validate_code("exit()\nprint(1)")
    assert not ok


def test_sandbox_blocks_os_system():
    from core.sandbox import validate_code

    ok, err = validate_code("import os\nos.system('dir')")
    assert not ok


def test_sandbox_allows_numpy_experiment():
    from core.sandbox import execute_sandboxed

    code = """
import numpy as np
import json
np.random.seed(0)
x = np.random.randn(100)
metrics = {"mean": float(x.mean()), "std": float(x.std())}
print(json.dumps({"metrics": metrics}))
"""
    result = execute_sandboxed(code, seed=0)
    assert result["success"], result.get("error")
    assert "metrics" in result.get("parsed", {})


def test_multi_seed_aggregation():
    from core.sandbox import run_multi_seed

    code = """
import numpy as np, json, random
x = [random.random() for _ in range(50)]
print(json.dumps({"metrics": {"acc": float(sum(x)/len(x))}}))
"""
    out = run_multi_seed(code, n_seeds=3)
    assert out["success"]
    assert out["n_success"] == 3
    assert "acc" in out["aggregate_metrics"]
    assert "mean" in out["aggregate_metrics"]["acc"]
    assert "std" in out["aggregate_metrics"]["acc"]


# ---------------------------------------------------------------------------
# 2. Citation hard checks
# ---------------------------------------------------------------------------

def test_extract_citation_ids():
    from core.verification import extract_citation_ids

    text = "See doi:10.1038/nature14539 and arXiv:1706.03762 (Vaswani et al., 2017)."
    ids = extract_citation_ids(text)
    assert any("10.1038" in d for d in ids["dois"])
    assert "1706.03762" in ids["arxiv_ids"]


def test_verify_citations_fake_doi(monkeypatch):
    from core import verification

    def fake_resolve(doi):
        return {"resolved": False, "doi": doi, "error": "not found"}

    monkeypatch.setattr(verification, "resolve_doi", fake_resolve)
    result = verification.verify_citations("Claim supported by doi:10.9999/fake.doi.123")
    assert result["passed"] is False
    assert result["score"] < 10


def test_verify_citations_real_doi_mocked(monkeypatch):
    from core import verification

    monkeypatch.setattr(
        verification,
        "resolve_doi",
        lambda doi: {"resolved": True, "doi": doi, "title": "Attention Is All You Need"},
    )
    result = verification.verify_citations("As shown in doi:10.5555/3295222.3295349")
    assert result["passed"] is True
    assert result["score"] == 10.0


# ---------------------------------------------------------------------------
# 3. Statistical validity
# ---------------------------------------------------------------------------

def test_statistical_validity_match():
    from core.verification import verify_statistics

    raw = {
        "aggregate_metrics": {
            "accuracy": {"mean": 0.85, "std": 0.02, "values": [0.83, 0.85, 0.87], "n": 3}
        }
    }
    reported = {"metrics": {"accuracy": {"mean": 0.85, "std": 0.02}}}
    result = verify_statistics(reported, raw_data=raw)
    assert result["passed"]
    assert result["score"] == 10.0


def test_statistical_validity_mismatch():
    from core.verification import verify_statistics

    raw = {
        "aggregate_metrics": {
            "accuracy": {"mean": 0.85, "std": 0.02, "values": [0.83, 0.85, 0.87], "n": 3}
        }
    }
    reported = {"metrics": {"accuracy": {"mean": 0.99, "std": 0.01}}}  # fabricated
    result = verify_statistics(reported, raw_data=raw)
    assert result["passed"] is False
    assert len(result["mismatches"]) >= 1


# ---------------------------------------------------------------------------
# 4. Planner falsifiability
# ---------------------------------------------------------------------------

def test_contribution_requires_falsifiable_prediction():
    from agents.planner import PlannerAgent

    planner = PlannerAgent.__new__(PlannerAgent)
    bad = {
        "expected_contributions": ["We improve accuracy somehow"],
        "contributions": [
            {
                "claim": "Better accuracy",
                # missing falsifiable_prediction and statistical_test
            }
        ],
    }
    flags = PlannerAgent._flag_unfalsifiable(planner, bad)
    assert len(flags) >= 1


def test_plan_requires_baselines():
    from agents.planner import PlannerAgent

    planner = PlannerAgent.__new__(PlannerAgent)
    plan = {
        "experiments": [
            {
                "name": "Main",
                "baselines": [],
                "baseline_comparison": "",
            }
        ]
    }
    missing = PlannerAgent._flag_missing_baselines(planner, plan)
    assert len(missing) >= 1


# ---------------------------------------------------------------------------
# 5. Code-claim consistency (heuristic)
# ---------------------------------------------------------------------------

def test_code_claim_consistency_mismatch():
    from agents.engineer import EngineerAgent

    eng = EngineerAgent.__new__(EngineerAgent)
    method_text = "We implement Gradient Boosting with XGBoost for classification."
    code = "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()"
    result = EngineerAgent.check_code_claim_consistency(eng, method_text, code)
    assert result["consistent"] is False or result["score"] < 8


# ---------------------------------------------------------------------------
# 6. Cross-run memory
# ---------------------------------------------------------------------------

def test_cross_run_memory(tmp_path):
    from core.run_log import CrossRunMemory

    mem = CrossRunMemory(path=str(tmp_path / "cross.jsonl"))
    mem.record_rejection("topic", "Saturated GAN topic", "novelty too low", {"sim": 0.95})
    lessons = mem.lessons_for_prompt()
    assert "Saturated GAN" in lessons
    assert "REJECTED" in lessons


# ---------------------------------------------------------------------------
# 7. LLM provider config
# ---------------------------------------------------------------------------

def test_config_resolve_model_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    # Reload is heavy; test the method on a fresh-ish config object
    from core.config import Config

    c = Config(
        llm_provider="openai",
        openai_model="gpt-4o-mini",
        gemini_model="gemini-2.5-flash",
    )
    assert c.resolve_model("default") == "gpt-4o-mini"


def test_apply_runtime_keys():
    from core.config import config, apply_runtime_keys

    apply_runtime_keys({"RESEARCH_DOMAIN": "biology"})
    assert config.research_domain == "biology"
    apply_runtime_keys({"RESEARCH_DOMAIN": "computer_science"})


# ---------------------------------------------------------------------------
# 8. Debate multi-round structure
# ---------------------------------------------------------------------------

def test_debate_result_has_rounds_field():
    from agents.hypothesis_debate import DebateResult

    fields = DebateResult.__dataclass_fields__
    assert "rounds" in fields or "objections" in fields or "elo_delta" in fields


def test_research_pipeline_streams_node_outputs():
    from core.pipeline import ResearchPipeline

    class FakeApp:
        def stream(self, source, config):
            assert source == {"current_phase": "start"}
            assert config["configurable"]["thread_id"] == "run-1"
            yield {"node_a": {"current_phase": "done"}}

    class FakeGraph:
        def compile(self, checkpointer):
            assert checkpointer == "checkpoint"
            return FakeApp()

    pipeline = ResearchPipeline(lambda: FakeGraph(), lambda: "checkpoint")
    assert list(pipeline.stream({"current_phase": "start"}, "run-1")) == [
        ("node_a", {"current_phase": "done"})
    ]


def test_results_writer_only_keeps_engineer_numbers(monkeypatch):
    """The post-engineering pass must reject fabricated quantitative claims."""
    import main

    class FakeWriter:
        def draft_section(self, section, *_args):
            return "# Results\nAccuracy was 85.0% with standard deviation 0.02."

    monkeypatch.setattr(main, "WriterAgent", FakeWriter)
    state = main.initialize_state()
    state.update({
        "selected_topic": {"title": "Test", "description": "test"},
        "plan": {"sections": ["Results"]},
        "engineer_outputs": {"experiment": {"aggregate_metrics": {"accuracy": {"mean": 0.85, "std": 0.02}}}},
    })
    result = main.write_results_sections(state)
    check = result["results_verification"]["Results"]
    assert check["passed"], check
    assert result["current_phase"] == "supervision"


def test_results_number_verifier_flags_untraced_claim():
    from main import verify_result_numbers

    result = verify_result_numbers("The model achieved 99.0% accuracy.", {"x": {"accuracy": 0.85}})
    assert not result["passed"]
    assert result["mismatches"]


def test_elo_breaks_topic_ranking_tie(monkeypatch, tmp_path):
    from agents import topic_hunter

    elo_path = tmp_path / "elo.json"
    elo_path.write_text('{"attention": 1700, "graph": 1300}', encoding="utf-8")
    monkeypatch.setattr(topic_hunter.config, "elo_ratings_path", str(elo_path))
    monkeypatch.setattr(topic_hunter, "call_llm", lambda *_a, **_k: '{"ranked_topics":[{"original_index":0,"rank":1,"score":8},{"original_index":1,"rank":1,"score":8}]}')
    agent = topic_hunter.TopicHunterAgent.__new__(topic_hunter.TopicHunterAgent)
    ranked = agent.rank_topics_by_potential([
        {"title": "Attention hypothesis", "feasibility": 7, "gap_score": 5},
        {"title": "Graph hypothesis", "feasibility": 7, "gap_score": 5},
    ])
    assert ranked[0]["hypothesis_kind"] == "attention"


def test_topic_hunter_uses_current_arxiv_client_api(monkeypatch):
    from agents.topic_hunter import TopicHunterAgent

    class Result:
        title, summary, entry_id, categories = "A", "Abstract", "id", ["cs.AI"]
        class published: year = 2026
        authors = []
    class Client:
        def __init__(self, **_kwargs): pass
        def results(self, _search): return iter([Result()])
    monkeypatch.setattr("agents.topic_hunter.arxiv.Client", Client)
    hunter = TopicHunterAgent.__new__(TopicHunterAgent)
    hunter.source_health = {}
    assert hunter.search_arxiv("test", 1)[0]["title"] == "A"
    assert hunter.source_health["arxiv"]["ok"]


def test_source_outage_finishes_without_reset(monkeypatch):
    import main
    from agents.topic_hunter import ResearchSourceUnavailable

    class Hunter:
        def discover_topics(self, _domain):
            raise ResearchSourceUnavailable("OpenAlex and arXiv unavailable")
    monkeypatch.setattr(main, "TopicHunterAgent", Hunter)
    state = main.initialize_state()
    out = main.topic_discovery_node(state)
    assert out["current_phase"] == "complete"
    assert out["terminal_error"] == "OpenAlex and arXiv unavailable"
    assert not out["should_reset"]


def test_research_ledger_stores_claims_and_events(tmp_path):
    from core.research_db import ResearchDatabase

    db = ResearchDatabase(str(tmp_path / "ledger.sqlite"))
    db.record_event("phase_change", {"phase": "planning"}, "r1", "orchestrator")
    db.create_run("r1", "2026-01-01T00:00:00Z")
    db.record_scratch("r1", "writer", "draft", {"text": "hello"}, {})
    db.record_artifact("r1", "raw_results", "output/raw.json", {})
    db.finish_run("r1", "completed", "editing", {"ok": True})
    db.record_claim("r1", "Results", "accuracy 0.85", "quantitative", "verified", {"metric": "accuracy"})
    assert db.recent_events("r1")[0]["data"]["phase"] == "planning"
    assert db.claims("r1")[0]["status"] == "verified"
    import sqlite3
    with sqlite3.connect(tmp_path / "ledger.sqlite") as con:
        assert con.execute("SELECT status FROM research_runs WHERE run_id='r1'").fetchone()[0] == "completed"
        assert con.execute("SELECT count(*) FROM run_scratchpad").fetchone()[0] == 1


def test_reproducibility_dossier_requires_executable_artifacts():
    from core.verification import reproducibility_dossier

    plan = {"experiments": [{"falsifiable_prediction": "A > B", "baselines": ["B"], "statistical_test": "t-test"}]}
    complete = {"x": {"raw_results_path": "raw.json", "code": "print(1)"}}
    assert reproducibility_dossier(plan, complete)["passed"]
    assert not reproducibility_dossier(plan, {})["passed"]



# ---------------------------------------------------------------------------
# 9. Eval harness meta — ensure tests package is the measurement surface
# ---------------------------------------------------------------------------

def test_eval_harness_exists():
    assert Path(__file__).exists()
    assert (ROOT / "core" / "sandbox.py").exists()
    assert (ROOT / "core" / "verification.py").exists()
    assert (ROOT / "core" / "llm.py").exists()
