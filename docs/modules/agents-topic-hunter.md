# `agents/topic_hunter.py`

## Responsibility

Discovers candidate research topics from OpenAlex, arXiv, and Semantic Scholar, enriches them with citation-graph signals, checks novelty and feasibility, and records rejected candidates. Periodically forces exploration of under-observed or lower-rated hypothesis kinds without bypassing gates.

## Main API

`TopicHunterAgent.discover_topics()` is the orchestration entry point. `TopicHunterAgent.retrieve_literature(query)` backs the QA-mode graph. Search, graph, novelty, feasibility, and rejection helpers are kept on the agent today.

## v3 changes

- **arXiv via official client**: `search_arxiv` uses the shared `arxiv.Client` (1 req / 3s pacing, Retry-After-aware retries). The old portalocker `/tmp` file lock is gone. Gateway still paces + breaks.
- **Adaptive gateway**: see `core/api_gateway.py` — AIMD buckets, jitter, single-flight coalescing.
- **LLM seed generation**: `_generate_llm_seeds` (strategy `llm_diverse`) with static fallback; flag `LLM_SEED_GENERATION_ENABLED` (default on).
- **Filler-word query filtering**: `_SEED_FILLER_WORDS` keeps meta-vocabulary out of search keywords.
- **Two-pool budget**: generation cannot starve the gate chain (reserved sub-budget).
- **Relevance-ranked evidence**: `_rank_papers_for_gap` selects the closest prior work for screener/novelty prompts.
- **retrieve_literature**: multi-source QA retrieval (previously missing — QA runs crashed).

## Cross-run context

Consumes `CrossRunMemory.get_prompt_context()` — structured rejection/pivot/revision tags only, never raw free-text reasons.

## Exploration

Every `topic_exploration_every` rankings, `_apply_exploration` may promote an under-observed kind (`selection_mode=forced_exploration`). Feasibility, novelty, and evidence gates still apply to explored candidates.

## Dependencies

External scholarly APIs (all through `core.api_gateway`), embeddings, LLM JSON generation, FAISS memory (write-only narrative), and structured cross-run lessons. Source outages are surfaced as `ResearchSourceUnavailable` when no usable source remains.
