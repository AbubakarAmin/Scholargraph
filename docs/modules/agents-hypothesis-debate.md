# `agents/hypothesis_debate.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Runs adversarial proposal and critique rounds, then uses one or more judge models to score the hypothesis. Records unresolved objections and updates coarse hypothesis-kind Elo ratings with observation counts and prior shrinkage.

## Main API

`HypothesisDebateSystem.conduct_debate()` returns a `DebateResult` dataclass. `ProposerAgent`, `ChallengerAgent`, `ModeratorAgent`, and `EloStore` are supporting components.

## v4 upgrade (2026-09)

- **Evidence-grounded proposer**: Round-1 arguments include structured hypothesis, retrieved evidence, and prior objection tags to preempt recurring objections.

## v4.1 fixes (2026-09-19, from run logs)

- **Objection normalization** (`_normalize_objection_payload`): Bare array / single-dict objections → canonical envelope instead of crashing.
- **Self-correcting parse** (`build_rebuttal`): `call_llm_json` re-ask carries the parse error + offending excerpt.
- **Followup envelope tolerance** (`followup_objections`): Bare-array objection payloads are coerced instead of failed closed.
- **Iteration hardening** (`evaluate_debate`, `conduct_debate`, `revise_topic_from_objections`): All `.get`-on-objection loops filter `isinstance(o, dict)`; `structured_hypothesis` accessed via `_as_dict()`.

## Memory integrity

- Full debate transcripts (proposer/challenger prose) are retained for audit.
- Future debates retrieve only structured tags: `objection_type`, `severity`, `resolution_status`.
- Prior argument text is never injected into new debate prompts.
- `EloStore` persists `{rating, raw_rating, observations, prior, shrinkage_k, updated_at}` (legacy flat floats upgrade on load). Selection uses the shrunk rating.

## Quality rule

A high mean score is insufficient when judge disagreement is high or severe objections remain unresolved.
