# `agents/hypothesis_debate.py`

## Responsibility

Runs adversarial proposal and critique rounds, then uses one or more judge models to score the hypothesis. It records unresolved objections and updates coarse hypothesis-kind Elo ratings.

## Main API

`HypothesisDebateSystem.conduct_debate()` returns a `DebateResult` dataclass. `ProposerAgent`, `ChallengerAgent`, `ModeratorAgent`, and `EloStore` are supporting components.

## Quality rule

A high mean score is insufficient when judge disagreement is high or severe objections remain unresolved.
