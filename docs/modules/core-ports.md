# Ports — `core/ports.py`

Persistence port protocols for future adapter injection.

## Purpose

Defines `Protocol` classes that specify the interface contracts for persistence backends. This enables future migration from the current SQLite/FAISS implementation to alternative backends without changing agent code.

## Protocols

### `EventPort`
```python
def record_event(event_type, data, run_id=None, agent=None) -> None
```

### `ResearchLedgerPort(EventPort)`
Extends `EventPort` with:
```python
def create_run(run_id, started_at) -> None
def finish_run(run_id, status, phase, summary) -> None
def record_claim(run_id, section, claim, claim_type, status, evidence) -> None
def record_artifact(run_id, artifact_type, location, metadata) -> None
```

### `VectorMemoryPort`
```python
def add_embedding(embedding, metadata) -> None
def get_prompt_context(query_embedding=None, *, k=5, namespace=None, ...) -> List[Dict]
def search_similar(query_embedding, k=5) -> List[Dict]
def audit_search_similar(query_embedding, k=5) -> List[Dict]
def add_debate_entry(...) -> None
def add_feedback_entry(...) -> None
def get_feedback_signals(agent_name=None, limit=10) -> List[Dict]
```

### `ArtifactPort`
```python
def save(name, content) -> str
```

## Current implementations

- `ResearchLedgerPort` → `core.research_db.ResearchDatabase`
- `VectorMemoryPort` → `core.memory.FAISSMemory`
- `EventPort` → also `ResearchDatabase` (events table)
