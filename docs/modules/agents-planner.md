# `agents/planner.py`

## Responsibility

Turns an accepted topic into a falsifiable research plan with paper sections, contributions, experiments, baselines, variants, metrics, dependencies, and a timeline.

## Main API

- `create_plan(topic)` builds and validates a new plan.
- `revise_plan(plan, revision_request, topic)` handles Engineer-to-Planner recovery.
- Internal flagging methods detect missing predictions, tests, and baselines.

## Contract

Every contribution should have a falsifiable prediction and statistical test. Every experiment should name a real baseline.
