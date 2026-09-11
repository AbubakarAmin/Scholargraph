## Module Overview

SupervisorAgent — hard deterministic checks first, LLM soft review last.
Citation grounding + statistical validity gate soft scores.

# `agents/supervisor.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Scores paper sections using hard deterministic checks first, then math/code checks and soft peer review. It records claim evidence in the research ledger.

## Main API

`SupervisorAgent.evaluate_section()` returns `(score, feedback)`. `MathChecker`, `CodeChecker`, and `ReviewerBot` implement focused checks.

## Memory integrity

Raw review prose is stored for human audit. Prompt-facing retrieval uses structured verdicts only (`verdict`, `score`, `blocking`, `category`, `section`). MetaAgent reads `get_feedback_signals()`, never raw `get_recent_feedback()`.

## Quality rule

A hard citation or statistics failure caps the section score and cannot be rescued by an LLM reviewer.
