"""Persistence ports for future adapter injection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class EventPort(Protocol):
    def record_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        run_id: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> None: ...


class ResearchLedgerPort(EventPort, Protocol):
    def create_run(self, run_id: str, started_at: str) -> None: ...

    def finish_run(
        self,
        run_id: str,
        status: str,
        phase: str,
        summary: Dict[str, Any],
    ) -> None: ...

    def record_claim(
        self,
        run_id: str,
        section: str,
        claim: str,
        claim_type: str,
        status: str,
        evidence: Dict[str, Any],
    ) -> None: ...

    def record_artifact(
        self,
        run_id: str,
        artifact_type: str,
        location: str,
        metadata: Dict[str, Any],
    ) -> None: ...


class VectorMemoryPort(Protocol):
    def add_embedding(self, embedding: Any, metadata: Dict[str, Any]) -> None: ...

    def search_similar(self, query_embedding: Any, k: int = 5) -> List[Dict[str, Any]]: ...

    def add_debate_entry(
        self,
        topic: str,
        proposer_argument: str,
        challenger_argument: str,
        moderator_decision: str,
        score: float,
    ) -> None: ...

    def add_feedback_entry(
        self,
        agent_name: str,
        section: str,
        score: float,
        feedback: str,
        iteration: int,
    ) -> None: ...


class ArtifactPort(Protocol):
    def save(self, name: str, content: Any) -> str: ...
