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
python main.py --mode qa --query "your question"  # QA literature synthesis mode
```

## Key gotchas

- After changing any backend Python file, **restart** `run_ui.py` — Uvicorn does not auto-reload this app.
- `.env` is loaded from project root (not cwd) via `dotenv` in `core/config.py:14`.
- `memory/keys.json` stores UI-entered API keys. Never commit it; it's in `.gitignore`.
- `memory/checkpoints.sqlite` and `memory/research_ledger.sqlite` are durable run state. Both are gitignored.
- Matplotlib is forced to `Agg` backend process-wide (`core/config.py:18`) to avoid Tcl crashes on Windows.
- `run_ui.py` kills existing processes on the same port before binding — no manual cleanup needed.
- Config defaults in `core/config.py` may differ from `env_example.txt` (e.g., `DEBATE_PASS_THRESHOLD` is `7.0` in code, `7.5` in env_example.txt). Code wins at runtime.
- `logs/` contains per-run log files (`<run_id>.log`) and a global rotating `scholargraph.log`. All errors are logged with full tracebacks via `exc_info=True`.

## Project structure

This is a flat Python project, not a monorepo.

| Directory | Purpose |
|---|---|
| `agents/` | One file per agent role (topic_hunter, debate, planner, writer, engineer, data, execution, analysis, verification, supervisor, meta_agent, editor) |
| `core/` | Shared services: config, LLM client, sandbox, memory, verification, workflow graph, research DB, source broker, contracts, capabilities, **api_gateway** |
| `web/` | FastAPI app (`app.py`) + static frontend (`static/admin.html`) |
| `tests/` | Offline eval harness and per-module tests (all mocked/offline) |
| `output/` | Generated papers, raw results, events, companion repo (gitignored) |
| `memory/` | FAISS index, keys, cross-run lessons, Elo ratings (gitignored) |
| `templates/` | LaTeX templates |

Root-level `test_*.py` files are standalone scripts, not part of the pytest suite.

## API Gateway (core/api_gateway.py)

All external API calls (arXiv, OpenAlex, Semantic Scholar, LLM, embeddings) are routed through a centralized `APIGateway` singleton. This provides:

- **Token bucket rate limiting** per provider (thread-safe, no sleep-outside-lock bugs)
- **Circuit breakers** per provider (trips after N consecutive failures, auto-recovers)
- **Retry with exponential backoff** (3 retries, 3s base, 30s max)
- **Source health tracking** (success rate per provider over sliding window)

### Rate limits (from official docs)

| Provider | Rate | Burst | Source |
|----------|------|-------|--------|
| arXiv | 0.33 req/s | 1 | 1 req / 3s (official) |
| OpenAlex | 5.0 req/s | 3 | 10 req/s polite pool |
| Semantic Scholar | 1.0 req/s | 2 | 1 RPS with API key |
| LLM | 0.5 req/s | 1 | Provider-dependent |
| Embedding | 1.0 req/s | 1 | Same as LLM |

### Usage

```python
from core.api_gateway import get_gateway

gateway = get_gateway()
result = gateway.request("arxiv", search_fn, query, max_results=100)
```

### Configuration

Rate limits are hardcoded in `APIGateway.DEFAULT_RATES`. Override via constructor:

```python
gateway = APIGateway(rates={"arxiv": (0.5, 2), "llm": (1.0, 1)})
```

### Adaptive behavior (v2)

- Rates are **starting points**. Every 429/503 penalizes that provider's `AdaptiveTokenBucket` (rate ×0.5, floor 0.05/s); after 5 consecutive successes the rate recovers additively toward base (AIMD).
- All backoff sleeps carry ±20% jitter.
- `gateway.request(..., coalesce_key=...)` single-flights identical concurrent read-only calls.
- `gateway.is_available(provider)` lets callers skip a provider whose breaker is open instead of paying retry sleeps.
- arXiv calls go through the official `arxiv` client (shared singleton, `delay_seconds=3.0`, `num_retries=3`) wrapped by the gateway — do NOT reintroduce raw HTTP or sleep-based pacing for arXiv.

## Topic Hunter Architecture

The topic hunter (`agents/topic_hunter.py`) is the most complex agent. Key design decisions:

### Sequential seed processing
Seeds are processed **sequentially** (not in parallel threads) to avoid overwhelming external APIs. The `seed_sequential` config flag (default `true`) controls this.

### Persona ensemble (sequential)
Persona calls (skeptic, practitioner) run **sequentially** inside each seed, not in sub-threads. The `persona_sequential` config flag (default `true`) controls this.

### Embedding precomputation
Abstract embeddings are computed **once** per seed (for up to `novelty_max_abstracts` papers) and reused across all gap novelty evaluations. This reduces embedding calls from ~198/seed to ~20/seed.

### LLM budget: two pools (v3)
Each seed is capped at `llm_budget_per_seed` (default 30) LLM calls, split into **generation** (persona/seed prompts) and **gate-chain** pools. Generation must leave a reserve (`min(max(6, budget//4), 12)`) so generated gaps are always evaluated instead of silently discarded. Non-numeric budget values fall back to 30.

### Seed & query quality (v3)
- `_generate_llm_seeds` mints specific technical seeds (strategy `llm_diverse`) from cross-run lessons; falls back to static template seeds on failure. Flag: `LLM_SEED_GENERATION_ENABLED`.
- `_SEED_FILLER_WORDS` ("gaps", "problems", "open", ...) are stripped before building arXiv/OpenAlex queries — searching for meta-vocabulary retrieved noise.
- `literature_evidence` is relevance-ranked per gap (`_rank_papers_for_gap`), not "first 8 fetched".

### QA-mode retrieval
`TopicHunterAgent.retrieve_literature(query)` backs the QA-mode graph (OpenAlex + arXiv + S2 bulk, deduped). It previously did not exist — QA runs crashed.

### Graceful degradation
If some sources fail but others succeed, partial results are returned instead of throwing `ResearchSourceUnavailable`. Open breakers short-circuit to a skip via `is_available()`.

## Run order that matters

1. `validate_config()` runs at startup — requires provider key + `OPENALEX_EMAIL`
2. Tests are offline/mocked and don't need API keys
3. Lint/typecheck: no configured tooling — the project uses plain Python without ruff/mypy/black enforcement
4. `SANDBOX_BACKEND` defaults to `docker` — if the Docker daemon is unreachable, dispatch falls back to the AST sandbox automatically (one warning). Set `SANDBOX_BACKEND=ast` explicitly for in-process execution.

## Testing

```bash
python -m pytest tests/ -q              # full suite
python -m pytest tests/test_eval_harness.py -q  # eval harness only
python -m pytest tests/test_api_gateway.py -q   # gateway tests
python tests/smoke_offline.py           # quick standalone smoke
```

The eval harness (`tests/test_eval_harness.py`) is the primary measurement surface for "did this upgrade help." Add tests there before wiring new gates.

## Architecture in one sentence

LangGraph orchestrates a pipeline: Topic Hunter → Debate → Planner → Data Validation → Writer (pre-engineering) → Engineer → Independent Validation (Execution → Analysis → Verification) → Writer (post-engineering) → Supervisor → Editor.

Graph definition lives in `core/workflow.py` and `core/workflow_nodes.py`. State is `ResearchState` TypedDict in `core/state.py`.

A separate QA-mode graph (`core/workflow.py:create_qa_graph`) runs: Literature Retrieval → Synthesis Answer → Verification. Invoked via `--mode qa --query "..."`.

## Config fields (core/config.py)

### API gateway settings
| Field | Default | Env Var | Purpose |
|-------|---------|---------|---------|
| `seed_sequential` | `true` | `SEED_SEQUENTIAL` | Process seeds sequentially (vs thread pool) |
| `persona_sequential` | `true` | `PERSONA_SEQUENTIAL` | Process personas sequentially (vs sub-threads) |
| `llm_budget_per_seed` | `30` | `LLM_BUDGET_PER_SEED` | Max LLM calls per seed thread |
| `novelty_max_abstracts` | `20` | `NOVELTY_MAX_ABSTRACTS` | Max abstracts for embedding precomputation |

### LLM settings
| Field | Default | Env Var | Purpose |
|-------|---------|---------|---------|
| `LLM_MAX_RETRIES` | `3` | `LLM_MAX_RETRIES` | Max retries for LLM calls (app-level) |
| `LLM_RETRY_BACKOFF` | `2.0` | `LLM_RETRY_BACKOFF` | Base backoff seconds for LLM retries |

## What NOT to do

- Don't assume the capability broker (`core/capabilities.py`) is a security boundary — it's application-level policy, not OS isolation.
- Don't treat the sandbox (`core/sandbox.py`) as container-level isolation — it's AST + restricted builtins in-process.
- Don't trust a visually complete manuscript — quantitative claims must pass `verify_statistics` and citation resolution before supervision.
- Don't commit `memory/`, `output/`, `.env`, or `*.sqlite*` files.
- Don't add `ThreadPoolExecutor` for external API calls — use the gateway's rate limiter and process sequentially.
- Don't bypass the gateway for arXiv/OpenAlex/S2/LLM calls — always use `gateway.request(provider, fn, ...)`.

## v4 research upgrades (2026-09)

Informed by 2025–2026 agentic-research literature (WARA artifact repair, AppliedScientist reviewer-guided revision, AgentGrad failure gradients, Gupta & Pruthi ACL 2025 novelty-plagiarism findings, TruthInsightBench "discriminating acts"):

| Upgrade | Where | Behavior |
|---|---|---|
| Feedback-aware revision | `agents/writer.py` (`draft_section(..., revision_feedback)`), `core/workflow_nodes.py:section_revision_feedback` | Redrafts carry deterministic check failures + supervisor feedback into the writer prompt; results redrafts re-draft only failing sections |
| Narrative revision loop | `write_narrative_sections` | On meta-continue, below-threshold narrative sections are re-drafted once with supervisor feedback (`narrative_revision_count` bound = 1) |
| Editor referee repair | `workflow_nodes.editor_repair_route` + `editing_node` | Release-referee failures route to one bounded repair pass instead of terminal failure; failed-experiment failures stay terminal |
| Self-correcting JSON | `core/utils.py:call_llm_json(call_fn=...)` | Re-asks with parse error on malformed JSON; wired into `consistency_referee`, supervisor soft checks, meta feedback |
| Novelty-plagiarism gate | `core/verification.py:novelty_overlap_check` | Deterministic overlap-coefficient screen of Abstract+Introduction vs closest prior work / literature evidence; wired into `final_manuscript_referee(sections, plan, outputs, topic=...)` |
| Evidence-grounded proposer | `agents/hypothesis_debate.py:ProposerAgent` | Round-1 arguments include structured hypothesis, retrieved evidence, and prior objection tags to preempt recurring objections |
| Prompt hardening | `agents/writer.py` | Intro/abstract get literature evidence + "never invent citations" policy; Results gets copy-exact + n=/std/CI + statistical-test + falsifiability-reporting requirements |
| LLM failure visibility | `core/llm.py`, `core/run_log.py` | `llm_failures` run stat + error-level message on chat failure |

New state fields: `narrative_revision_count`, `editor_repair_count`, `editor_repair_findings` (core/state.py). Tests: `tests/test_v4_research_upgrades.py`.

### v4.1 round two (2026-09, same batch)

Informed by Anthropic context engineering (attention budget / minimal high-signal tokens), Critic Experience Bank, CYCLE, and AgentGrad:

| Upgrade | Where | Behavior |
|---|---|---|
| Engineer failure-gradient hints | `agents/engineer.py:_error_category_hint` | Deterministic hints per failure category (timeout → shrink workload; import → allowed imports; sandbox → no file/subprocess; JSON → metrics line) injected into `_refine_code` prompts |
| Cross-run lessons for Engineer | `_generate_experiment_code` | Uses the sanctioned `CrossRunMemory().get_prompt_context()` boundary (never `lessons_for_prompt` — the memory-integrity static audit forbids it) to list prior failure patterns |
| Screener self-correction | `agents/topic_hunter.py` research screener | `call_llm_json` replaces the parse-blind 2-attempt loop |
| Challenger followup self-correction | `agents/hypothesis_debate.py:followup_objections` | Same self-correcting parse; prior objections still preserved fail-closed |
| QA answer robustness | `core/workflow_nodes.py:qa_answer_node` | `call_llm_json` with parse-error re-ask |
| Supervisor checklist hardening | `agents/supervisor.py:REVIEW_CHECKLIST` | Requires explicit falsifiable-prediction verdict (supported/falsified/inconclusive) with controls/robustness evidence (TruthInsightBench gap) |
| Writer context budget | `agents/writer.py` | Experiment JSON dumps bounded (`[:6000]`) per context-rot guidance |

Candidate next upgrades (not yet implemented): GEPA-style cross-run prompt evolution over the prompt registry; HippoRAG-style graph memory for cross-run lessons; model cascades (cheap drafter → strong verifier) per FrugalGPT; workflow search (AFlow) over the debate/planning graph.

## Key wiring fixes (2026-09)

| Fix | Where | What |
|---|---|---|
| Editing repair route now live in graph | `core/workflow.py` | `add_edge("editing", END)` replaced with conditional edge routing to `writing_results` when `editor_repair_count >= 1` |
| Generic section via _draft_with_retry | `agents/writer.py:_draft_generic_section` | Was calling `call_llm` directly, bypassing revision feedback and min-char retry; now routes through `_draft_with_retry(section_name, prompt)` |
| Engineer memory-boundary compliance | `agents/engineer.py` | First attempt used forbidden `lessons_for_prompt()`; switched to sanctioned `CrossRunMemory().get_prompt_context()` |

## Debate payload-shape fixes (2026-09-19, from run logs)

| Fix | Where | What |
|---|---|---|
| `'list' object has no attribute 'get'` crash | `agents/hypothesis_debate.py:build_rebuttal` | `parse_json_from_llm` returns a **list** for bare JSON arrays; the challenger now coerces via `_normalize_objection_payload` (bare array / single-dict objections → canonical envelope) instead of crashing the whole subsystem |
| Challenger self-correcting parse | `build_rebuttal` | Replaced the parse-blind identical re-roll with `call_llm_json` (re-ask carries the parse error + offending excerpt); degenerate outputs feed the re-ask path |
| Followup envelope tolerance | `followup_objections` | Bare-array objection payloads are coerced instead of logged `degenerate_followup` and failed closed |
| Objection iteration hardening | `evaluate_debate`, `conduct_debate`, `revise_topic_from_objections` | All `.get`-on-objection loops filter `isinstance(o, dict)`; `structured_hypothesis` accessed via `_as_dict()` |
| `parse_json_from_llm` type safety | `core/utils.py` | Non-str input → None; broad `except Exception` (was JSONDecodeError-only) |

Tests: `tests/test_debate_robustness.py`.
