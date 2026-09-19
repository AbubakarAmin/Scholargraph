# Sources S2 Bulk

**File:** `core/sources_s2_bulk.py` (62 lines)

## Purpose

Semantic Scholar bulk search client — a free, no-API-key-required text search source for papers, used alongside OpenAlex and arXiv for literature discovery.

## Key functions

| Function | Purpose |
|---|---|
| `search_s2_bulk(query, headers, limit, year_from, fields)` | Queries the S2 bulk endpoint, returns normalized paper dicts |

## Return format

Each result is a dict with: `title`, `abstract`, `year`, `doi`, `arxiv_id`, `cited_by_count`, `source`.

## Configuration

- Requests are routed through `get_gateway().request("s2_bulk", ...)` with retry.
- Default fields: `title,abstract,year,externalIds,citationCount`.
- Limit is capped at 100 per the S2 API.

## Gotchas

- On any failure (network, 429, etc.), returns an empty list rather than raising — graceful degradation.
- 429 responses are converted to `RateLimitError` with the provider's `Retry-After` header.
- No API key required, but rate limits apply (1 req/s with key, lower without).
