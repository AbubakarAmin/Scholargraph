"""Regression tests for immutable experiment handoffs and terminal gates."""

import json

from core.evidence_gate import build_contract, gate_engineering_outputs, validate_dataset_identity, validate_experiments
from core.artifacts import save_results
from core.state import initialize_state
from core.verification import validate_empirical_claims
from core.workflow_nodes import engineering_node, independent_validation_node


def test_validate_experiments_rejects_null_and_string_candidates():
    assert validate_experiments([None])
    assert validate_experiments(["experiment"])
    assert validate_experiments([{"name": "toy", "variants": ["bad"]}])


def test_dataset_identity_requires_the_committed_artifact():
    experiment = {"name": "toy", "evaluation_metrics": ["accuracy"], "dataset": {"name": "fixed"}}
    assert validate_dataset_identity([experiment], {})
    assert not validate_dataset_identity(
        [experiment],
        {"fixed": {"content_hash": "abc", "validation": {"passed": True}}},
    )
    assert validate_dataset_identity(
        [{**experiment, "dataset": {"name": "fixed", "content_hash": "changed"}}],
        {"fixed": {"content_hash": "abc", "validation": {"passed": True}}},
    )


def test_engineering_gate_requires_successful_output_bound_to_contract(tmp_path):
    experiment = {"name": "toy", "evaluation_metrics": ["accuracy"]}
    contract = build_contract(experiment)
    raw = tmp_path / "toy.json"
    raw.write_text(json.dumps({"metrics": {"accuracy": 0.2}}), encoding="utf-8")

    decision = gate_engineering_outputs(
        {"experiments": [experiment]},
        {
            "toy": {
                "success": True,
                "contract_hash": contract["contract_hash"],
                "raw_results_path": str(raw),
                "aggregate_metrics": {"accuracy": {"mean": 0.2}},
            }
        },
        {"toy": contract},
    )

    assert decision["allowed"]
    assert not decision["terminal"]


def test_engineering_gate_stops_failed_output_even_with_error_artifact(tmp_path):
    experiment = {"name": "toy", "evaluation_metrics": ["accuracy"]}
    contract = build_contract(experiment)
    raw = tmp_path / "FAIL_toy.json"
    raw.write_text(json.dumps({"error": "syntax error"}), encoding="utf-8")

    decision = gate_engineering_outputs(
        {"experiments": [experiment]},
        {
            "toy": {
                "success": False,
                "contract_hash": contract["contract_hash"],
                "raw_results_path": str(raw),
                "error": "syntax error",
            }
        },
        {"toy": contract},
    )

    assert decision["terminal"]
    assert decision["reason_code"] == "technical_execution_failure"


def test_engineering_node_does_not_run_malformed_experiment(monkeypatch):
    state = initialize_state()
    state.update({
        "plan": {"experiments": [None]},
        "selected_topic": {"title": "test"},
    })

    class ExplodingEngineer:
        def __init__(self, *_args):
            raise AssertionError("engineer must not be created for malformed plans")

    monkeypatch.setattr("core.workflow_nodes.EngineerAgent", ExplodingEngineer)
    result = engineering_node(state)

    assert result["current_phase"] == "complete"
    assert not result["should_continue"]
    assert result["evidence_gate"]["reason_code"] == "invalid_experiment_plan"


def test_engineering_node_rejects_contract_drift(monkeypatch):
    original = {"name": "toy", "evaluation_metrics": ["accuracy"]}
    state = initialize_state()
    state.update({
        "plan": {"experiments": [{**original, "evaluation_metrics": ["f1"]}]},
        "selected_topic": {"title": "test"},
        "experiment_contracts": {"toy": build_contract(original)},
    })

    class ExplodingEngineer:
        def __init__(self, *_args):
            raise AssertionError("engineer must not run after contract drift")

    monkeypatch.setattr("core.workflow_nodes.EngineerAgent", ExplodingEngineer)
    result = engineering_node(state)

    assert result["current_phase"] == "complete"
    assert result["evidence_gate"]["reason_code"] == "experiment_contract_drift"


def test_independent_validation_stops_without_code_artifacts():
    state = initialize_state()
    state.update({
        "plan": {"experiments": [{"name": "toy", "evaluation_metrics": ["accuracy"]}]},
        "engineer_outputs": {
            "toy": {"success": False, "error": "runtime failure"},
        },
    })

    result = independent_validation_node(state)

    assert result["current_phase"] == "complete"
    assert not result["should_continue"]
    assert result["terminal_error"]


def test_terminal_run_writes_failure_dossier_without_latex(tmp_path):
    state = initialize_state()
    state.update({
        "run_id": "failed-run",
        "terminal_error": "sandbox runtime failure",
        "evidence_gate": {"terminal": True, "reason_code": "technical_execution_failure"},
        "latex_output": "\\section{Fabricated}",
    })

    save_results(state, str(tmp_path))

    assert (tmp_path / "failure_dossier.json").exists()
    assert not (tmp_path / "paper_output.tex").exists()


def test_empirical_claim_check_rejects_harness_diagnostics():
    result = validate_empirical_claims(
        "The run failed because tracemalloc was blocked by the sandbox.",
        {"toy": {"success": True}},
    )

    assert not result["passed"]
    assert "tracemalloc" in result["prohibited_text"]
