"""Offline tests for brokered source retrieval and replay."""

import json

import pytest
import requests

from core.sources import SourceClient, SourcePolicy


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status
        self.content = json.dumps(payload).encode("utf-8")

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_source_client_fetches_validates_hashes_and_caches(tmp_path):
    session = FakeSession([FakeResponse({"results": [{"title": "A paper"}]})])
    client = SourceClient(str(tmp_path), session=session)

    first = client.fetch_json(
        "openalex",
        "https://api.openalex.org/works",
        params={"search": "reproducibility"},
        validator=lambda payload: "results" in payload,
    )
    second = client.fetch_json(
        "openalex",
        "https://api.openalex.org/works",
        params={"search": "reproducibility"},
        validator=lambda payload: "results" in payload,
    )

    assert first["status"] == "verified"
    assert second["status"] == "cached"
    assert first["response_hash"] == second["response_hash"]
    assert len(session.calls) == 1


def test_source_client_retries_and_returns_explicit_unavailable(tmp_path):
    session = FakeSession([
        requests.ConnectionError("offline"),
        requests.ConnectionError("offline"),
    ])
    client = SourceClient(str(tmp_path), session=session, policy=SourcePolicy(retries=1))

    result = client.fetch_json("openalex", "https://api.openalex.org/works")

    assert result["status"] == "unavailable"
    assert result["warnings"]
    assert len(session.calls) == 2


def test_source_client_rejects_non_allowlisted_urls(tmp_path):
    client = SourceClient(str(tmp_path))

    with pytest.raises(PermissionError):
        client.fetch_json("openalex", "http://api.openalex.org/works")

    with pytest.raises(PermissionError):
        client.fetch_json("openalex", "https://example.com/works")
