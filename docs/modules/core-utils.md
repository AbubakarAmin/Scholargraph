# `core/utils.py`

## Responsibility

Shared non-provider helpers: JSON parsing, citation extraction, embeddings, math checks, similarity, and action logging. Provider access belongs to `core.llm`.

## Refactor note

This module remains a broad utility bucket for compatibility, but it no longer re-exports LLM provider setup or call aliases. New code should prefer narrowly named services such as `core.llm` and `core.verification`.
