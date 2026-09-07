# `agents/hypothesis_debate.py`

## Responsibility

Runs adversarial proposal and critique rounds, then uses one or more judge models to score the hypothesis. It records unresolved objections and updates coarse hypothesis-kind Elo ratings with observation counts and prior shrinkage.

## Main API

`HypothesisDebateSystem.conduct_debate()` returns a `DebateResult` dataclass. `ProposerAgent`, `ChallengerAgent`, `ModeratorAgent`, and `EloStore` are supporting components.

## Memory integrity

- Full debate transcripts (proposer/challenger prose) are retained for audit.
- Future debates retrieve only structured tags: `objection_type`, `severity`, `resolution_status`.
- Prior argument text is never injected into new debate prompts.
- `EloStore` persists `{rating, raw_rating, observations, prior, shrinkage_k, updated_at}` (legacy flat floats upgrade on load). Selection uses the shrunk rating.

## Quality rule

A high mean score is insufficient when judge disagreement is high or severe objections remain unresolved.
