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


def test_verify_citations_rejects_resolved_but_mismatched_metadata(monkeypatch):
    from core import verification

    monkeypatch.setattr(
        verification,
        "resolve_doi",
        lambda doi: {"resolved": True, "doi": doi, "title": "Actual Paper", "authors": ["Gulrajani"]},
    )
    result = verification.verify_citations(
        "doi:10.5555/3295222.3295349",
        {"10.5555/3295222.3295349": {"title": "Wasserstein GANs with Gradient Penalty", "authors": ["Korotin"]}},
    )
    assert not result["passed"]
    assert result["metadata_mismatches"]
    mismatch = result["metadata_mismatches"][0]
    assert mismatch["metadata_diff"]["title"]["expected"] == "Wasserstein GANs with Gradient Penalty"
    assert mismatch["metadata_diff"]["title"]["actual"] == "Actual Paper"


def test_verify_citations_requires_all_supplied_authors(monkeypatch):
    from core import verification

    monkeypatch.setattr(
        verification,
        "resolve_doi",
        lambda doi: {
            "resolved": True,
            "doi": doi,
            "title": "A Reliable Result",
            "authors": ["Smith", "Jones"],
        },
    )
    result = verification.verify_citations(
        "doi:10.5555/example",
        {"10.5555/example": {"title": "A Reliable Result", "authors": ["Smith", "Wrong"]}},
    )
    assert not result["passed"]
    assert result["metadata_mismatches"][0]["metadata_diff"]["authors"]["actual"] == ["Smith", "Jones"]


def test_extract_citation_metadata_reads_bibliography_fields():
    from core.verification import extract_citation_metadata

    metadata = extract_citation_metadata(
        '@article{x, title={Actual Title}, author={Smith and Jones}, doi={10.5555/example}}'
    )
    assert metadata["10.5555/example"]["title"] == "Actual Title"
    assert "Smith" in metadata["10.5555/example"]["authors"]


def test_cross_section_numeric_consistency_rejects_conflicting_metric_values():
    from core.verification import cross_section_numeric_consistency

    result = cross_section_numeric_consistency(
        {"Abstract": "Experiment toy accuracy was 80%.", "Results": "Experiment toy accuracy was 90%."},
        {"toy": {"aggregate_metrics": {"accuracy": {"mean": 0.8}}}},
    )
    assert not result["passed"]
    assert result["conflicts"][0]["metric"] == "accuracy"


def test_numeric_consistency_rejects_single_claim_against_artifact():
    from core.verification import cross_section_numeric_consistency

    result = cross_section_numeric_consistency(
        {"Results": "Experiment toy accuracy was 90%."},
        {"toy": {"aggregate_metrics": {"accuracy": {"mean": 0.8}}}},
    )
    assert not result["passed"]
    assert result["conflicts"][0]["reason"] == "manuscript claim disagrees with structured artifact"
    assert any(claim["sections"] == ["Results"] for claim in result["claims"])


def test_consistency_referee_blocks_model_reported_contradiction(monkeypatch):
    from core import verification

    monkeypatch.setattr(
        verification,
        "call_llm",
        lambda *_args, **_kwargs: '{"findings": [{"category": "dataset", "message": "Sections name different datasets", "blocking": true}]}',
    )
    result = verification.consistency_referee({"Methods": "Iris", "Results": "Digits"})
    assert not result["passed"]
    assert result["findings"][0]["category"] == "dataset"


def test_consistency_referee_malformed_output_blocks(monkeypatch):
    from core import verification

    monkeypatch.setattr(verification, "call_llm", lambda *_args, **_kwargs: "not json")
    result = verification.consistency_referee({"Results": "No contradictions"})
    assert not result["passed"]
    assert result["findings"][0]["category"] == "referee_error"


def test_capability_manifest_rejects_external_large_plan():
    from core.capabilities import check_plan_feasibility

    errors = check_plan_feasibility({"methodology": "download Yahoo Finance data with GPU training"})
    assert any("outbound" in error for error in errors)
    assert any("GPU" in error for error in errors)


def test_planner_rescopes_infeasible_plan():
    from agents.planner import PlannerAgent

    planner = PlannerAgent.__new__(PlannerAgent)
    plan = {
        "methodology": "GPU model with download",
        "experiments": [{"name": "x", "dataset": {"name": "Yahoo Finance"}, "evaluation_metrics": ["accuracy"]}],
    }
    planner._generate_plan_structure = lambda *_args: plan
    planner._ensure_falsifiable_contributions = lambda current, _topic: []
    planner._generate_experiments = lambda *_args: plan["experiments"]
    planner._attach_variants = lambda experiments: experiments
    planner._flag_unfalsifiable = lambda *_args: []
    planner._flag_missing_baselines = lambda *_args: []
    planner._generate_dependencies = lambda *_args: []
    planner._generate_timeline = lambda *_args: []
    planner._store_plan = lambda *_args: None
    planner.context = type("Context", (), {"config": type("Runtime", (), {"research_domain": "test"})()})()
    topic = {"title": "Yahoo Finance forecasting", "description": "production market forecasting"}
    result = planner.create_plan(topic)
    assert result["capability_rescope"]
    assert result["experiments"][0]["dataset"]["name"] == "bundled_synthetic"
    assert "bundled benchmark" in result["title"]
    assert "bounded local" in result["abstract"]
    assert "external-domain" in topic["description"]


def test_analysis_adds_multiple_comparison_adjustment():
    from agents.analysis import AnalysisAgent

    comparisons = [{"p_value": 0.03}, {"p_value": 0.04}]
    AnalysisAgent._apply_multiple_comparison_correction(comparisons, {})
    assert comparisons[0]["p_value_adjusted"] == 0.06
    assert comparisons[0]["multiple_comparison_policy"] == "bonferroni"


def test_debate_challenger_adds_manifest_feasibility_objection(monkeypatch):
    from agents.hypothesis_debate import ChallengerAgent

    monkeypatch.setattr(
        "agents.hypothesis_debate.call_llm",
        lambda *_args, **_kwargs: '{"objections": [], "summary_rebuttal": "ok"}',
    )
    challenger = ChallengerAgent.__new__(ChallengerAgent)
    challenger.context = None
    challenger.client = None
    challenger.build_rebuttal(
        {"title": "External", "description": "download Yahoo Finance data with GPU training", "dataset_plan": "Yahoo Finance"},
        "proposal",
    )
    assert any(item["criterion"] == "feasibility" and item["severity"] == 5 for item in challenger._last_objections)


def test_writer_does_not_serialize_failed_engineer_diagnostics():
    from agents.writer import WriterAgent

    writer = WriterAgent.__new__(WriterAgent)
    prompt_data = writer._format_engineer_outputs({"toy": {"success": False, "error": "tracemalloc import blocked"}})
    assert "tracemalloc" not in prompt_data
    assert "failure_summaries" in prompt_data
    assert "sandbox_validation" in prompt_data
    assert "experiments" in prompt_data


def test_known_answer_check_blocks_wrong_generated_result():
    from core.sandbox import run_known_answer_check

    result = run_known_answer_check(
        'import json\nprint(json.dumps({"metrics": {"eigenvalue": 2.0}}))',
        {"eigenvalue": 1.0},
    )
    assert not result["passed"]
    assert "eigenvalue" in result["mismatches"]


def test_tournament_keeps_highest_scoring_survivor(monkeypatch):
    from agents.hypothesis_debate import DebateResult, HypothesisDebateSystem

    system = HypothesisDebateSystem.__new__(HypothesisDebateSystem)
    # First fails, second passes — tournament must stop before later candidates.
    outcomes = {
        "a": DebateResult("a", "", "", "FAIL", 4.0, False, ""),
        "Spectral Invariance for Dynamic Permutation Sets": DebateResult(
            "Spectral Invariance for Dynamic Permutation Sets", "", "", "PASS", 7.5, True, ""
        ),
        "c": DebateResult("c", "", "", "FAIL", 5.0, False, ""),
    }
    called = []

    def conduct(topic):
        called.append(topic["title"])
        return outcomes[topic["title"]]

    system.conduct_debate = conduct
    results = system.conduct_tournament(
        [
            {"title": "a", "score": 4},
            {"title": "Spectral Invariance for Dynamic Permutation Sets", "score": 7.5},
            {"title": "c", "score": 5},
        ],
        rounds=2,
    )
    assert results[0].topic == "Spectral Invariance for Dynamic Permutation Sets"
    assert results[0].passed
    assert "c" not in called
    assert called == ["a", "Spectral Invariance for Dynamic Permutation Sets"]


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
    experiment = {
        "name": "Model Comparison",
        "baselines": ["gradient_boosting", "xgboost"],
        "claimed_components": [],
    }
    code = "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()"
    result = EngineerAgent.check_code_claim_consistency(eng, experiment, code)
    assert result["consistent"] is False or result["score"] < 8


def test_code_claim_consistency_ignores_free_text_method_prose():
    from agents.engineer import EngineerAgent

    eng = EngineerAgent.__new__(EngineerAgent)
    experiment = {
        "name": "RF only",
        "baselines": ["random_forest"],
        "claimed_components": [],
    }
    # Free-text "neural network"/"svm" must not create false failures.
    code = (
        "# commentary: we considered neural network and svm but implemented RF\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "model = RandomForestClassifier()\n"
    )
    result = EngineerAgent.check_code_claim_consistency(eng, experiment, code)
    assert result["consistent"] is True
    assert result["notes"] == []


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
    hunter = TopicHunterAgent.__new__(TopicHunterAgent)
    hunter.source_health = {}
    hunter._arxiv_client = Client(page_size=100, delay_seconds=3.0, num_retries=3)
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


def test_research_ledger_reconstructs_claim_lineage(tmp_path):
    from core.research_db import ResearchDatabase

    db = ResearchDatabase(str(tmp_path / "lineage.sqlite"))
    db.create_run("r1", "2026-01-01T00:00:00Z")
    db.record_artifact("r1", "raw", "raw.json", {"artifact_id": "a1"})
    db.record_claim("r1", "Results", "accuracy 0.8", "empirical_result", "verified", {"artifact_ids": ["a1"]})
    lineage = db.claim_lineage("r1")
    assert lineage[0]["artifacts"][0]["location"] == "raw.json"


def test_reproducibility_dossier_requires_executable_artifacts(tmp_path):
    from core.verification import reproducibility_dossier

    plan = {"experiments": [{"falsifiable_prediction": "A > B", "baselines": ["B"], "statistical_test": "t-test"}]}
    raw = tmp_path / "raw.json"
    raw.write_text("{}", encoding="utf-8")
    complete = {"x": {"raw_results_path": str(raw), "code": "print(1)", "contract_hash": "abc"}}
    assert reproducibility_dossier(plan, complete)["passed"]
    assert not reproducibility_dossier(plan, {})["passed"]
    empty = reproducibility_dossier(None, None)
    assert empty["checks"]["contract_provenance"] is False
    assert empty["checks"]["limitations_disclosed"] is False
    assert not empty["passed"]


# ---------------------------------------------------------------------------
# 9. Eval harness meta — ensure tests package is the measurement surface
# ---------------------------------------------------------------------------

def test_eval_harness_exists():
    assert Path(__file__).exists()
    assert (ROOT / "core" / "sandbox.py").exists()
    assert (ROOT / "core" / "verification.py").exists()
    assert (ROOT / "core" / "llm.py").exists()
