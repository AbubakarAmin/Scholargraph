## Module Overview

SupervisorAgent — hard deterministic checks first, LLM soft review last.
Citation grounding + statistical validity gate soft scores.

# `agents/supervisor.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Scores paper sections using hard deterministic checks first, then math/code checks and soft peer review. Records claim evidence in the research ledger.

## Main API

`SupervisorAgent.evaluate_section()` returns `(score, feedback)`. `MathChecker`, `CodeChecker`, and `ReviewerBot` implement focused checks.

## Order (critical)

1. Hard citation + stats (`hard_verify_section`)
2. MathChecker (SymPy parse + optional equality checks)
3. CodeChecker (`compile` / balanced delimiters)
4. If hard fail → soft review still runs but **overall score capped at 4.0**
5. If hard pass → ReviewerBot (LLM) + soft hallucination pass

Weighted blend when hard passed: hard 0.35 + math 0.15 + code 0.15 + reviewer 0.20 + hall 0.15.

## v4 upgrade (2026-09)

- **Checklist hardening** (`REVIEW_CHECKLIST`): Requires explicit falsifiable-prediction verdict (supported/falsified/inconclusive) with controls/robustness evidence (TruthInsightBench gap).

## Memory integrity

Raw review prose is stored for human audit. Prompt-facing retrieval uses structured verdicts only (`verdict`, `score`, `blocking`, `category`, `section`). MetaAgent reads `get_feedback_signals()`, never raw `get_recent_feedback()`.

## Quality rule

A hard citation or statistics failure caps the section score and cannot be rescued by an LLM reviewer.
