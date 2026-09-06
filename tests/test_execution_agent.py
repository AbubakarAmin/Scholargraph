"""Offline tests for independent reproducible execution."""

import json

from agents.execution import ExecutionAgent


def test_execution_agent_records_seeded_results_and_hash(tmp_path):
    agent = ExecutionAgent.__new__(ExecutionAgent)
    agent.output_dir = tmp_path
    code = 'import json\nprint(json.dumps({"metrics": {"accuracy": 0.8}}))'

    artifact = agent.execute(
        {"experiment_name": "toy", "source_code": code},
        {"experiment_name": "toy", "seeds": [7, 8]},
    )

    assert artifact["status"] == "completed"
    assert artifact["environment"]["seeds"] == [7, 8]
    assert artifact["content_hash"]
    assert json.loads(open(artifact["raw_results_path"], encoding="utf-8").read())["n_seeds"] == 2


def test_execution_agent_rejects_forbidden_code(tmp_path):
    agent = ExecutionAgent.__new__(ExecutionAgent)
    agent.output_dir = tmp_path

    artifact = agent.execute(
        {"experiment_name": "bad", "source_code": "import subprocess\nsubprocess.run([])"},
        {"experiment_name": "bad", "seeds": [1]},
    )

    assert artifact["status"] == "failed"
    assert "rejected" in artifact["error"].lower()