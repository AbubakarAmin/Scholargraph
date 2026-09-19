# API Gateway

**File:** `core/api_gateway.py` (507 lines)

## Purpose

Centralized gateway for all external API calls (arXiv, OpenAlex, Semantic Scholar, LLM, embeddings). Provides adaptive rate limiting, circuit breaking, retry with jitter, and single-flight request coalescing.

## Key classes

| Class | Purpose |
|---|---|
| `TokenBucket` | Thread-safe fixed-rate token bucket rate limiter |
| `AdaptiveTokenBucket(TokenBucket)` | AIMD-style rate adaptation: cuts rate on 429/503, recovers additively after 5 consecutive successes |
| `CircuitBreaker` | CLOSED/OPEN/HALF_OPEN state machine; opens after N consecutive failures, auto-recovers after cooldown |
| `SourceHealthTracker` | Per-provider success rates over a sliding window |
| `APIGateway` | Main entry point. `request(provider, fn, ...)` applies rate limiting, circuit breaking, retry, jitter |

## Usage

```python
from core.api_gateway import get_gateway

gateway = get_gateway()
result = gateway.request("arxiv", search_fn, query, max_results=100)
```

## Rate limits

| Provider | Rate | Burst | Source |
|----------|------|-------|--------|
| arXiv | 0.33 req/s | 1 | 1 req / 3s (official) |
| OpenAlex | 5.0 req/s | 3 | 10 req/s polite pool |
| Semantic Scholar | 1.0 req/s | 2 | 1 RPS with API key |
| LLM | 0.5 req/s | 1 | Provider-dependent |
| Embedding | 1.0 req/s | 1 | Same as LLM |

Rate limits are hardcoded in `APIGateway.DEFAULT_RATES`. Override via constructor:

```python
gateway = APIGateway(rates={"arxiv": (0.5, 2), "llm": (1.0, 1)})
```

## Adaptive behavior

- Rates are **starting points**. Every 429/503 penalizes that provider's `AdaptiveTokenBucket` (rate ×0.5, floor 0.05/s); after 5 consecutive successes the rate recovers additively toward base (AIMD).
- All backoff sleeps carry ±20% jitter.
- `gateway.request(..., coalesce_key=...)` single-flights identical concurrent read-only calls.
- `gateway.is_available(provider)` lets callers skip a provider whose breaker is open instead of paying retry sleeps.
- arXiv calls go through the official `arxiv` client (shared singleton, `delay_seconds=3.0`, `num_retries=3`) wrapped by the gateway — do NOT reintroduce raw HTTP or sleep-based pacing for arXiv.

## Gotchas

- 429 errors trigger a multiplicative rate penalty (x0.5, floor 0.05 req/s) plus a cooldown window — no fast retry.
- Transient errors are detected by substring matching on the exception message (e.g., "524", "timeout", "connection reset").
- Unknown providers get a default rate of 0.5 req/s with burst 1 (lazy bucket/breaker creation).
- `get_gateway()` returns a module-level singleton (double-checked locking). Use `reset_gateway()` in tests.
