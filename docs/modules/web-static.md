# `web/static/admin.html`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Single-page Control Deck frontend. It polls the FastAPI endpoints and renders pipeline progress, configuration, paper drafts, debate results, experiments, plan checks, and evidence trace.

## Backend contract

The page depends on `/api/health`, `/api/config`, `/api/keys`, `/api/dashboard`, `/api/events`, `/api/scratchpad`, `/api/artifacts`, `/api/capabilities`, and `/api/run/status`.

## UI behavior

- **Provider-aware configuration**: choosing Gemini shows only Gemini credentials; choosing OpenAI shows OpenAI credentials; choosing OpenAI-compatible also requires a base URL.
- **Operations-first home**: release readiness, current phase, experiment count, blocking findings, gate checks, and latest activity are the first view.
- **Live refresh**: the browser polls status/dashboard/scratchpad every 3 s.
- **Error presentation**: a stopped run displays a visible, actionable error banner.
- `index.html` remains as a legacy fallback.
