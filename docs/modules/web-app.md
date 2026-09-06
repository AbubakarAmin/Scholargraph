# `web/app.py`

## Responsibility

FastAPI backend for the Control Deck. It exposes health, configuration, key management, run status, dashboard, events, scratchpad, memory, and run deletion endpoints.

## Runtime model

`POST /api/run` starts the same LangGraph pipeline used by the CLI in a daemon thread. The latest state is held in process memory while durable events and evidence are written through core services.

## Operational warning

CORS is permissive and the API has no authentication. Bind to localhost for the current desktop deployment.
