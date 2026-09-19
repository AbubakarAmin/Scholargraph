# Sources OpenReview

**File:** `core/sources_openreview.py` (192 lines)

## Purpose

OpenReview client that fetches venue submissions and extracts reviewer-stated weaknesses from Official_Review replies as pre-formalized research gap signals.

## Key classes

| Class/Function | Purpose |
|---|---|
| `OpenReviewClient` | Wraps the `openreview-py` v2 API client. Requires username/password authentication. |
| `fetch_venue_gap_signals(venue_key, max_submissions)` | Paginates through submissions, extracts weaknesses from reviews, returns gap signal dicts |
| `get_openreview_client()` | Module-level singleton accessor; reads credentials from config |

## Return format

Each result is a dict with: `title`, `abstract`, `weaknesses`, `venue`, `year`, `source`.

## Configuration

- `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` in `.env`
- `OPENREVIEW_VENUES` for venue keys (default: `["iclr2025", "iclr2026", "neurips2025"]`)
- `OPENREVIEW_ENABLED` to toggle (default: `true`)

## Gotchas

- `editdistance` (transitive dep of `openreview-py`) requires a C++ compiler; on Python 3.14+ where wheels are missing, a stub module is injected so the API client can still load.
- Wrong venue invitation strings silently return zero results (not an error — a common pitfall).
- Only submissions with at least one non-empty weakness are returned.
- Pagination uses `get_notes` (not `get_all_notes`) to avoid fetching all 11k+ submissions.
- Auth failures disable the client gracefully (returns empty lists).
- OpenReview API v2 (`api2.openreview.net`) is required; v1 is legacy.
