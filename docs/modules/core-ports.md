## Module Overview

Persistence ports for future adapter injection.

# `core/ports.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Defines Protocol interfaces for event recording, research-ledger persistence, vector memory, and artifact storage.

## Migration role

The existing SQLite, JSONL, FAISS, and filesystem implementations remain compatible adapters. New code can depend on these ports rather than concrete global stores.
