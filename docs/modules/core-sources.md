# `core/sources.py`

`SourceClient` caches allowlisted JSON and text responses with hashes and provenance. `fetch_open_access_text()` requires an explicit license or open-access signal. Unknown, closed, or paywalled sources return `unavailable` and are never silently promoted to evidence.