## Module Overview

Shared utilities and common operations.
Use core.llm for LLM provider access and compatibility aliases.

# `core/utils.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Shared non-provider helpers: JSON parsing, citation extraction, embeddings, math checks, similarity, and action logging. Provider access belongs to `core.llm`.

## Key API

- `parse_json_from_llm(text)`: Extracts JSON from LLM output. Non-str input → None; broad `except Exception` (was JSONDecodeError-only). Used throughout debate and planning.
- `call_llm_json(call_fn, prompt, ...)`: Self-correcting JSON parse — re-asks with parse error on malformed JSON. Wired into `consistency_referee`, supervisor soft checks, meta feedback, screener, challenger followup, and QA answer node.
- `validate_math_expression`, `verify_math_derivation`: SymPy-based math validation.
- `log_agent_action`: Writes structured entries to the event stream.

## v4.1 upgrade

- **`call_llm_json`**: Replaces the parse-blind 2-attempt loops in topic hunter screener, challenger followup, and QA answer node. Carries the parse error + offending excerpt in the re-ask prompt.

## Refactor note

This module remains a broad utility bucket for compatibility, but it no longer re-exports LLM provider setup or call aliases. New code should prefer narrowly named services such as `core.llm` and `core.verification`.
