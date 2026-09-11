# `web/static/index.html`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Single-page Control Deck frontend. It polls the FastAPI endpoints and renders pipeline progress, configuration, paper drafts, debate results, experiments, plan checks, and evidence trace.

## Backend contract

The page depends on `/api/health`, `/api/config`, `/api/keys`, `/api/dashboard`, `/api/events`, `/api/scratchpad`, `/api/runs`, and `/api/run/status`.
