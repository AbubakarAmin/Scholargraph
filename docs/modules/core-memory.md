# `core/memory.py`

## Responsibility

Persists FAISS embeddings, metadata, debate history, and supervisor feedback.
Prompt retrieval is **fail-closed**: only `structured_signal` entries that are
explicitly `retrieval_eligible` may enter prompts, and only through
`get_prompt_context()`.

## Key API

| Method | Role |
|--------|------|
| `add_embedding` | Store vector + normalized metadata (legacy → `generated_narrative`, ineligible) |
| `get_prompt_context` | **Central prompt boundary** — structured signals only |
| `search_similar` | Prompt-safe similarity (strips narrative); prefer `get_prompt_context` |
| `audit_search_similar` | Audit/forensics only — returns raw metadata including prose |
| `add_debate_entry` / `add_feedback_entry` | Persist raw text for audit + optional structured `signal` |
| `get_feedback_signals` | Scores/verdicts for MetaAgent — never review prose |
| `get_recent_feedback` / `get_recent_debates` | Audit-only full rows |

## Prompt policy

- `content_class=generated_narrative` is never prompt-eligible (including Writer drafts, fallbacks, errors).
- Writer exemplars require `namespace=writer_exemplars`, `retrieval_eligible=True`, and `outcome_status=released`.
- Debate prompts may retrieve `objection_type`, `severity`, `resolution_status` only — never prior arguments.
- Legacy unclassified entries load successfully but stay ineligible until explicitly reclassified.

## Coupling

The module creates a global `memory` instance at import time and reads paths from global config. Agents must not call `search_similar`, `get_recent_feedback`, `get_recent_debates`, or `lessons_for_prompt` for prompt construction (enforced by `tests/test_memory_integrity.py`).
