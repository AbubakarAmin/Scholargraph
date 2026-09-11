## Module Overview

Typed handoff contracts shared by agents and workflow state.

# `core/contracts.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Defines TypedDict handoff contracts for topics, plans, experiment specifications and outputs, verification reports, and final papers.

## Design rule

Use these shapes at agent boundaries. Internal implementation details can remain flexible, but cross-phase data should not silently change shape.
