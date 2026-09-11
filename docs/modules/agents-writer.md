## Module Overview

WriterAgent - Drafts paper sections and integrates citations and code outputs.
Handles abstract, introduction, methods, results, and conclusion sections.

# `agents/writer.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Drafts individual paper sections from topic and plan context, optionally incorporating engineer outputs. It also stores section history in memory and provides fallback text when generation fails.

## Workflow usage

The orchestrator calls the writer twice: narrative sections before engineering, then result-bearing sections after engineering. The second pass is checked by numeric grounding in `main.py`.

## Refactor note

Provider-specific legacy helpers remain here. Migrate this module to `core.llm.call_llm` during compatibility cleanup.
