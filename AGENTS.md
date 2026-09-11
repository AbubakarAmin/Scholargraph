# AGENTS.md — ScholarGraph

## Quick reference

```bash
pip install -r requirements.txt
copy env_example.txt .env        # Windows; cp on macOS/Linux
python -m pytest tests -q        # full offline test suite
python tests/smoke_offline.py    # quick smoke check, no API keys
python run_ui.py                 # web UI at http://127.0.0.1:8765
python main.py                   # CLI run
python main.py --resume <run_id> # resume crashed run
```

## Key gotchas

- After changing any backend Python file, **restart** `run_ui.py` — Uvicorn does not auto-reload this app.
- `.env` is loaded from project root (not cwd) via `dotenv` in `core/config.py:14`.
- `memory/keys.json` stores UI-entered API keys. Never commit it; it's in `.gitignore`.
- `memory/checkpoints.sqlite` and `memory/research_ledger.sqlite` are durable run state. Both are gitignored.
- Matplotlib is forced to `Agg` backend process-wide (`core/config.py:18`) to avoid Tcl crashes on Windows.

## Project structure

This is a flat Python project, not a monorepo.

| Directory | Purpose |
|---|---|
| `agents/` | One file per agent role (topic_hunter, debate, planner, writer, engineer, data, execution, analysis, verification, supervisor, meta_agent, editor) |
| `core/` | Shared services: config, LLM client, sandbox, memory, verification, workflow graph, research DB, source broker, contracts, capabilities |
| `web/` | FastAPI app (`app.py`) + static frontend (`static/admin.html`) |
| `tests/` | Offline eval harness and per-module tests |
| `output/` | Generated papers, raw results, events, companion repo (gitignored) |
| `memory/` | FAISS index, keys, cross-run lessons, Elo ratings (gitignored) |
| `templates/` | LaTeX templates |

Root-level `test_*.py` files are standalone scripts, not part of the pytest suite.

## Run order that matters

1. `validate_config()` runs at startup — requires provider key + `OPENALEX_EMAIL`
2. Tests are offline/mocked and don't need API keys
3. Lint/typecheck: no configured tooling — the project uses plain Python without ruff/mypy/black enforcement

## Testing

```bash
python -m pytest tests/ -q              # full suite
python -m pytest tests/test_eval_harness.py -q  # eval harness only
python tests/smoke_offline.py           # quick standalone smoke
```

The eval harness (`tests/test_eval_harness.py`) is the primary measurement surface for "did this upgrade help." Add tests there before wiring new gates.

## Architecture in one sentence

LangGraph orchestrates a pipeline: Topic Hunter → Debate → Planner → Data Validation → Writer (pre-engineering) → Engineer → Independent Validation (Execution → Analysis → Verification) → Writer (post-engineering) → Supervisor → Editor.

Graph definition lives in `core/workflow.py` and `core/workflow_nodes.py`. State is `ResearchState` TypedDict in `core/state.py`.

## What NOT to do

- Don't assume the capability broker (`core/capabilities.py`) is a security boundary — it's application-level policy, not OS isolation.
- Don't treat the sandbox (`core/sandbox.py`) as container-level isolation — it's AST + restricted builtins in-process.
- Don't trust a visually complete manuscript — quantitative claims must pass `verify_statistics` and citation resolution before supervision.
- Don't commit `memory/`, `output/`, `.env`, or `*.sqlite*` files.
