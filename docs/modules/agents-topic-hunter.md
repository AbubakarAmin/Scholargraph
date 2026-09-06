# `agents/topic_hunter.py`

## Responsibility

Discovers candidate research topics from OpenAlex and arXiv, enriches them with citation-graph signals, checks novelty and feasibility, and records rejected candidates.

## Main API

`TopicHunterAgent.discover_topics()` is the orchestration entry point. Search, graph, novelty, feasibility, and rejection helpers are kept on the agent today.

## Dependencies

External scholarly APIs, embeddings, LLM JSON generation, FAISS memory, and cross-run lessons. Source outages are surfaced as `ResearchSourceUnavailable` when no usable source remains.
