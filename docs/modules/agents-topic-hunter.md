# `agents/topic_hunter.py`

## Responsibility

Discovers candidate research topics from OpenAlex and arXiv, enriches them with citation-graph signals, checks novelty and feasibility, and records rejected candidates. Periodically forces exploration of under-observed or lower-rated hypothesis kinds without bypassing gates.

## Main API

`TopicHunterAgent.discover_topics()` is the orchestration entry point. Search, graph, novelty, feasibility, and rejection helpers are kept on the agent today.

## Cross-run context

Consumes `CrossRunMemory.get_prompt_context()` — structured rejection/pivot/revision tags only, never raw free-text reasons.

## Exploration

Every `topic_exploration_every` rankings, `_apply_exploration` may promote an under-observed kind (`selection_mode=forced_exploration`). Feasibility, novelty, and evidence gates still apply to explored candidates.

## Dependencies

External scholarly APIs, embeddings, LLM JSON generation, FAISS memory (write-only narrative), and structured cross-run lessons. Source outages are surfaced as `ResearchSourceUnavailable` when no usable source remains.
