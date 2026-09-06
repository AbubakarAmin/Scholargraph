# `agents/supervisor.py`

## Responsibility

Scores paper sections using hard deterministic checks first, then math/code checks and soft peer review. It records claim evidence in the research ledger.

## Main API

`SupervisorAgent.evaluate_section()` returns `(score, feedback)`. `MathChecker`, `CodeChecker`, and `ReviewerBot` implement focused checks.

## Quality rule

A hard citation or statistics failure caps the section score and cannot be rescued by an LLM reviewer.
