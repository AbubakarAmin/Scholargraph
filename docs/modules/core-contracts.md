# `core/contracts.py`

## Responsibility

Defines TypedDict handoff contracts for topics, plans, experiment specifications and outputs, verification reports, and final papers.

## Design rule

Use these shapes at agent boundaries. Internal implementation details can remain flexible, but cross-phase data should not silently change shape.
