# `core/memory.py`

## Responsibility

Persists FAISS embeddings, metadata, debate history, and supervisor feedback.

## Key API

`ResearchMemory.add_embedding`, `search_similar`, `add_debate_entry`, `add_feedback_entry`, score queries, and `clear_all`.

## Coupling

The module creates a global `memory` instance at import time and reads paths from global config. A future service boundary should make storage injectable and avoid import-time I/O.
