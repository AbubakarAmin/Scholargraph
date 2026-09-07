"""Regression tests for run review fixes (error attribution, plan revision gating, downstream guards, variant schema, tournament deduplication)."""

from typing import Any, Dict, List
import pytest

from agents.hypothesis_debate import DebateResult, HypothesisDebateSystem
from agents.planner import PlannerAgent
from core.contracts import Topic
from core.evidence_gate import validate_experiments
from core.run_log import build_run_summary, CrossRunMemory
from core.state import initialize_state
from core.workflow import create_research_graph
from core.workflow_nodes import (
    data_validation_node,
    engineering_node,
    hypothesis_debate_node,
    independent_validation_node,
    planning_node,
    should_continue,
    should_reset,
    write_narrative_sections,
    write_results_sections,
)


def test_planner_normalizes_string_variants_to_objects():
    planner = PlannerAgent.__new__(PlannerAgent)
    raw_experiments = [
        {
            "name": "Circuit Fidelity Benchmark",
            "evaluation_metrics": ["accuracy"],
            "baselines": ["logistic_regression"],
            "variants": ["variant string A", "variant string B", "variant string C"],
        },
        {
            "name": "Logical Representation Analysis",
            "evaluation_metrics": ["f1"],
            "baselines": ["random_forest"],
            "variants": [
                {"variant_name": "Custom Variant 1", "methodology": "custom feature ablation"},
                "plain string variant",
            ],
        },
    ]
    normalized = planner._normalize_experiments(raw_experiments)
    assert len(normalized) == 2
    for exp in normalized:
        assert isinstance(exp["variants"], list)
        for variant in exp["variants"]:
            assert isinstance(variant, dict)
            assert "name" in variant and variant["name"]
    errors = validate_experiments(normalized)
    assert not errors


def test_planner_attach_variants_ensures_objects():
    planner = PlannerAgent.__new__(PlannerAgent)
    raw_experiments = [
        {
            "name": "Exp 1",
            "variants": ["string_var_1", "string_var_2"],
        }
    ]
    attached = planner._attach_variants(raw_experiments)
    assert len(attached) == 1
    assert len(attached[0]["variants"]) == 2
    for v in attached[0]["variants"]:
        assert isinstance(v, dict)
        assert "name" in v
        assert "methodology" in v
    errors = validate_experiments(attached)
    assert not errors


def test_planner_create_plan_repairs_schema_errors(monkeypatch):
    from agents import planner as planner_mod

    planner = PlannerAgent.__new__(PlannerAgent)
    planner.context = type("Context", (), {"config": type("Runtime", (), {"research_domain": "test"})()})()
    
    # Mock initial plan structure and generation
    planner._generate_plan_structure = lambda *_args: {
        "title": "Test Plan",
        "sections": [{"name": "Methods"}],
        "methodology": "local",
        "compute_budget": "cpu",
    }
    planner._ensure_falsifiable_contributions = lambda *_args: [{"claim": "c", "falsifiable_prediction": "p", "statistical_test": "t"}]
    # Initially returns invalid experiments (missing evaluation_metrics)
    planner._generate_experiments = lambda *_args: [{"name": "bad_exp"}]
    planner._flag_unfalsifiable = lambda *_args: []
    planner._flag_missing_baselines = lambda *_args: ["bad_exp has no baselines"]
    planner._generate_dependencies = lambda *_args: []
    planner._generate_timeline = lambda *_args: {}
    planner._store_plan = lambda *_args: None
    monkeypatch.setattr(planner_mod, "list_datasets", lambda: [])
    monkeypatch.setattr(planner_mod, "check_plan_feasibility", lambda *_args, **_kwargs: [])

    # Mock repair to produce valid experiments
    def fake_repair(plan, topic, unfalsifiable=None, missing_baselines=None, schema_errors=None):
        plan["experiments"] = [
            {
                "name": "repaired_exp",
                "baselines": ["baseline_a"],
                "evaluation_metrics": ["accuracy"],
                "falsifiable_prediction": "prediction",
                "statistical_test": "welch_t",
                "variants": [{"name": "v1", "methodology": "m1"}],
            }
        ]
        return plan

    planner._repair_plan = fake_repair

    plan = planner.create_plan({"title": "topic", "description": "desc"})
    assert plan["experiments"][0]["name"] == "repaired_exp"
    assert not validate_experiments(plan["experiments"])


def test_planning_node_terminal_failure_on_unrecoverable_schema_errors(monkeypatch):
    state = initialize_state()
    state["selected_topic"] = {"title": "Test Topic"}

    class FailingPlanner:
        def __init__(self, *_args):
            pass
        def create_plan(self, topic):
            return {
                "title": "Invalid Plan",
                "experiments": [{"invalid": "no name"}],
                "schema_revision_attempts": 3,
            }

    monkeypatch.setattr("core.workflow_nodes._create_agent", lambda cls: FailingPlanner())

    result = planning_node(state)
    assert result["current_phase"] == "complete"
    assert result["should_continue"] is False
    assert result["terminal_error"] is not None
    assert "contract validation" in result["terminal_error"] or "invalid_experiment_plan" in str(result.get("evidence_gate"))


def test_workflow_graph_routes_directly_to_end_on_planning_terminal_failure():
    nodes = {
        "topic_discovery": lambda s: s,
        "hypothesis_debate": lambda s: s,
        "planning": lambda s: s,
        "data_validation": lambda s: s,
        "writing_narrative": lambda s: s,
        "engineering": lambda s: s,
        "independent_validation": lambda s: s,
        "writing_results": lambda s: s,
        "supervision": lambda s: s,
        "meta_evaluation": lambda s: s,
        "editing": lambda s: s,
        "reset": lambda s: s,
        "should_reset": should_reset,
        "should_continue": should_continue,
    }
    graph = create_research_graph(nodes)

    state = initialize_state()
    state["selected_topic"] = {"title": "Test Topic"}
    state["current_phase"] = "complete"
    state["should_continue"] = False
    state["terminal_error"] = "Planning failed contract validation"

    executed_nodes = []
    def record_node(node_name):
        def _fn(s):
            executed_nodes.append(node_name)
            return s
        return _fn

    for name in nodes:
        if name not in ("should_reset", "should_continue"):
            nodes[name] = record_node(name)

    graph_with_recording = create_research_graph(nodes).compile()
    final_state = graph_with_recording.invoke(state)
    assert executed_nodes == ["topic_discovery"]


def test_downstream_nodes_do_not_overwrite_earliest_terminal_error():
    state = initialize_state()
    state["terminal_error"] = "Planning failed: Plan experiments failed contract validation"
    state["current_phase"] = "complete"
    state["should_continue"] = False

    # Call downstream nodes directly
    state = data_validation_node(state)
    assert state["terminal_error"] == "Planning failed: Plan experiments failed contract validation"

    state = write_narrative_sections(state)
    assert state["terminal_error"] == "Planning failed: Plan experiments failed contract validation"

    state = engineering_node(state)
    assert state["terminal_error"] == "Planning failed: Plan experiments failed contract validation"

    state = independent_validation_node(state)
    assert state["terminal_error"] == "Planning failed: Plan experiments failed contract validation"

    state = write_results_sections(state)
    assert state["terminal_error"] == "Planning failed: Plan experiments failed contract validation"


def test_conduct_tournament_debates_each_candidate_at_most_once(monkeypatch):
    debater = HypothesisDebateSystem.__new__(HypothesisDebateSystem)
    debater.context = None
    debater.runtime_config = None

    debated = []
    def fake_conduct_debate(topic):
        debated.append(topic["title"])
        return DebateResult(
            topic=topic["title"],
            proposer_argument="arg",
            challenger_argument="rebuttal",
            moderator_decision="FAIL",
            score=4.0,
            passed=False,
            reasoning="failed",
        )

    debater.conduct_debate = fake_conduct_debate
    topics = [
        {"title": "Topic 1"},
        {"title": "Topic 2"},
        {"title": "Topic 3"},
        {"title": "Topic 4"},
    ]

    results = debater.conduct_tournament(topics, rounds=2)
    assert len(results) == 4
    # All 4 debated exactly once, no repeated debates
    assert debated == ["Topic 1", "Topic 2", "Topic 3", "Topic 4"]


def test_run_summary_preserves_earliest_terminal_error():
    state = initialize_state()
    state["terminal_error"] = "Planning failed: Plan experiments failed contract validation"
    state["current_phase"] = "complete"
    state["meta_feedback"].append("Planning error: Plan experiments failed contract validation")

    summary = build_run_summary(state=state)
    assert summary["status"] == "failed"
    assert "Planning failed: Plan experiments failed contract validation" in summary["stopped_because"]
    assert "Planning failed: Plan experiments failed contract validation" in summary["narrative"]
