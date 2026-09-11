## Module Overview

Reliable, replayable access to allowlisted scholarly sources.

# `core/sources.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

`SourceClient` caches allowlisted JSON and text responses with hashes and provenance. `fetch_open_access_text()` requires an explicit license or open-access signal. Unknown, closed, or paywalled sources return `unavailable` and are never silently promoted to evidence.
