"""Contract tests for scoped agent capabilities."""

import pytest

from core.capabilities import DEFAULT_MANIFESTS, authorize, manifest_for
from core.tool_broker import CapabilityBroker


def test_engineer_cannot_execute_or_analyze():
    engineer = DEFAULT_MANIFESTS["EngineerAgent"]

    assert authorize(engineer, "code.generate").allowed
    assert not authorize(engineer, "code.execute").allowed
    assert not authorize(engineer, "analysis.statistics").allowed


def test_verifier_can_read_and_replay_but_not_change_code():
    verifier = DEFAULT_MANIFESTS["VerificationAgent"]

    assert authorize(verifier, "artifact.read").allowed
    assert authorize(verifier, "verification.replay").allowed
    assert not authorize(verifier, "code.generate").allowed


def test_unknown_capabilities_fail_closed():
    decision = authorize(DEFAULT_MANIFESTS["EngineerAgent"], "network.browser")

    assert not decision.allowed
    assert decision.reason == "unknown capability"


def test_manifest_rejects_unknown_or_overlapping_capabilities():
    with pytest.raises(ValueError, match="Unknown capabilities"):
        manifest_for("bad", "test", {"network.browser"})

    with pytest.raises(ValueError, match="both allowed and forbidden"):
        manifest_for("bad", "test", {"artifact.read"}, {"artifact.read"})


def test_broker_authorizes_and_audits_tool_calls():
    broker = CapabilityBroker({"read_artifact": lambda name: f"read:{name}"})

    result = broker.call(
        DEFAULT_MANIFESTS["VerificationAgent"],
        "artifact.read",
        "read_artifact",
        name="results.json",
    )

    assert result == "read:results.json"
    assert broker.calls[-1].status == "started"


def test_broker_denies_before_dispatching():
    called = []
    broker = CapabilityBroker({"execute": lambda: called.append(True)})

    with pytest.raises(PermissionError):
        broker.call(DEFAULT_MANIFESTS["EngineerAgent"], "code.execute", "execute")

    assert called == []
    assert broker.calls[-1].status == "denied"
