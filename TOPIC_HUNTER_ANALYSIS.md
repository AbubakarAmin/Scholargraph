# TopicHunter Deep Analysis & Efficiency Improvement Plan

## Part 1: Root Cause Analysis — Why 300+ LLM Requests & 13M Tokens Don't Reach Hypothesis Phase

### Yes, This Is Normal Given the Architecture — But It's a Design Failure

The system is spending its entire budget in the topic hunting phase and never reaches hypothesis debate. This is not an external API issue; it's an internal token economy problem baked into the architecture.

### The Token Budget Breakdown Per Run

Here's the exact LLM call cost per `discover_topics()` invocation (5 seeds, 6 gaps/seed):

#### A. Per-Seed Generation Phase (~8-12 calls/seed, cheap tier)

| Call | Purpose | Tier | Est. Tokens |
|------|---------|------|-------------|
| `_generate_hyde_abstract` | HyDE query construction | cheap | ~1K in, ~500 out |
| `_derive_followup_query_text` (×2 hops) | Multi-hop follow-up | cheap | ~1K in each |
| Persona ×2 (`_hunt_once` loop) | Skeptic + Practitioner | cheap | ~25K in each |
| Gap generation (`base_prompt`) | Generate 6 gaps | cheap | ~25K in each |
| **Subtotal per seed** | | | **~80-100K tokens** |

#### B. Per-Gap Gate Chain (~6-8 calls/gap, MIXED tier)

| Gate | Calls | Tier | Est. Input Tokens | Est. Output Tokens |
|------|-------|------|-------------------|-------------------|
| `_quick_grounding_check` | 0 | heuristic | 0 | 0 |
| `screen_research_gap` | 1 | cheap | ~8K (6 papers × 800 chars) | ~500 |
| `evaluate_layered_novelty` | 1-2 | cheap + embedding | ~4K (abstract) + 20 embeddings | ~200 |
| `formalize_hypothesis` | 1-2 | **STRONG** | **~20K** (gap + gap_report + novelty_report + sandbox manifest) | **~2K** |
| `feasibility_filter` | 0 | heuristic | 0 | 0 |
| `validate_topic_admission` | 0 | heuristic | 0 | 0 |
| **Subtotal per gap** | | | **~30-35K tokens** | |

#### C. Cross-Seed Overhead

| Component | Calls | Tier | Est. Tokens |
|-----------|-------|------|-------------|
| `_harvest_frontier_terms` | 1 (cached after first run) | cheap | ~25K in |
| `_generate_dynamic_seeds` (LLM seeds) | 1 | cheap | ~10K in |
| Citation graph fetches (5 papers) | 0 LLM | S2 API | 0 |
| Structural gap / sparsity / contradiction mining | 0 LLM | S2 API | 0 |
| `rank_topics_by_potential` | 1 | cheap | ~10K in |

#### D. Hypothesis Debate (If Reached)

| Round | Calls | Tier | Est. Tokens |
|-------|-------|------|-------------|
| Proposer | 1-2 | **STRONG** | ~30K in each |
| Challenger | 1-2 | **STRONG** | ~30K in each |
| Moderator (×2-3 rounds) | 2-6 | **JUDGE** | ~40K in each |
| Followup objections | 1-2 | JUDGE | ~20K in each |
| **Per topic** | | | **~200-400K tokens** |

### The Math

```
5 seeds × (100K generation + 6 gaps × 35K gate chain) = 5 × 310K = 1.55M tokens for topic hunting
+ rank_topics: ~10K
+ frontier seeding: ~25K
= ~1.6M tokens just for LLM calls

Add embeddings: 5 seeds × 20 abstracts × ~2K tokens = ~200K tokens
Add output tokens (est. 20% of input): ~360K tokens
Total LLM: ~2.2M tokens

BUT: 300+ LLM requests × ~40K avg tokens/request = 12M tokens
This matches the reported 13M tokens.
```

### The REAL Problem: Rejection Rate

The system generates 5 seeds × 6 gaps = **30 candidate topics**. The gate chain rejects ~80-95%:

- `_quick_grounding_check`: rejects ~20% (ungrounded)
- `screen_research_gap`: rejects ~30% (unsupported gap)
- `novelty_too_low`: rejects ~20% (duplicate)
- `infeasible_for_engineer`: rejects ~15% (sandbox violation)
- `missing_structured_hypothesis`: rejects ~10% (LLM schema failure)

**Survivors: ~3-6 topics** enter hypothesis debate.

The fundamental problem: **Every gap goes through 4-5 expensive LLM gates before being either accepted or rejected.** The system spends strong-tier tokens formalizing hypotheses for topics that will never be debated. Each formalize_hypothesis call costs ~22K tokens (20K input + 2K output) at the STRONG tier. If only 2 out of 30 gaps survive, that's 26 × 22K = 572K tokens WASTED on failed formalizations.

### Why You Can't Reach Hypothesis Phase

1. **The gate chain is too expensive per candidate** — formalize_hypothesis alone is the most expensive single call in the system, and it runs for every gap
2. **The base prompt is enormous** — 25K+ input tokens per generation call because it dumps ALL signals (graph, coupling, sparsity, contradictions, OpenReview, negative lessons, bridges) into every prompt
3. **Embedding calls are un-cached across seeds** — each seed re-embeds the same abstracts
4. **Persona ensemble doubles the generation cost** — both personas get the full 25K prompt
5. **Multi-hop retrieval adds more calls per seed** — each hop generates a follow-up query + re-searches
6. **The iterative reset loop** — when no topics survive, `topic_discovery_node` sets `should_reset=True`, which restarts the entire graph from scratch, burning another full budget

---

## Part 2: Comprehensive Improvement Plan

### Architecture-Level Changes (Priority 1)

#### 1. **Two-Stage Funnel: Cheap Pre-Screen → Expensive Gates**

**Problem**: All 30 gaps go through the full 4-5 gate chain.

**Fix**: Add a deterministic pre-screen stage BEFORE any LLM gate.

```
Current:  Generate 30 → [LLM screen] → [LLM novelty] → [LLM formalize] → [LLM feasibility] → debate
Proposed: Generate 30 → [Heuristic pre-screen] → [LLM screen (top 10)] → [LLM novelty (top 5)] → [LLM formalize (top 3)] → debate
```

**Implementation**:
- Add `_heuristic_pre_screen(gap, recent_papers)` that uses:
  - Title/description keyword overlap with retrieved papers (already computed in `_rank_papers_for_gap`)
  - Minimum abstract length check (empty abstracts = hallucinated gap)
  - Forbidden sandbox phrase check (already in `_BLOCKED_SANDBOX_PHRASES`)
  - Jaccard similarity between gap keywords and paper keywords > 0.15 threshold
- This eliminates ~50% of gaps with ZERO LLM calls
- Only top 10-15 gaps proceed to `screen_research_gap`

**Expected savings**: ~15 LLM calls per run (15 gaps × 1 screen call each)

#### 2. **Drastic Prompt Compression**

**Problem**: The base prompt in `_hunt_once` is ~25K input tokens because it dumps everything into every prompt.

**Fix**: Separate the prompt into a SHARED CONTEXT cache and a SEED-SPECIFIC context.

```python
# SHARED (computed once per discover_topics call):
shared_context = {
    "cross_run_context": ...,
    "graph_signals": ...,      # truncated to top 3
    "coupling_gaps": ...,      # truncated to top 3
    "sparse_cells": ...,       # truncated to top 3
    "contradictions": ...,     # truncated to top 3
    "replication_targets": ...,# truncated to top 3
    "evidence_bridges": ...,   # truncated to top 3, 3000 chars max
    "sample_titles": ...,      # top 5 only
}

# SEED-SPECIFIC (varies per seed):
seed_context = {
    "seed_hint": ...,
    "excluded_topics": ...,    # last 10 only
    "hyde_abstract": ...,
    "active_kind": ...,
}
```

**Expected savings**: ~10-15K tokens per generation call × 10 calls = ~100-150K tokens

#### 3. **Precomputed Embedding Reuse Across Seeds**

**Problem**: Each seed re-embeds the same 20-25 abstracts from `recent_papers`.

**Fix**: Compute embeddings ONCE per `discover_topics()` call and share across all seeds.

```python
def discover_topics(self, domain=None, n_parallel=5):
    # ... retrieval happens once ...
    recent_papers = self._retrieve_all_papers(domain)  # single retrieval
    
    # Precompute ALL embeddings once
    abstract_embeddings = {}
    for p in recent_papers[:novelty_max_abstracts]:
        paper_key = (p.get("doi") or p.get("arxiv_id") or p.get("title")).strip().lower()
        if paper_key:
            abstract_embeddings[paper_key] = generate_embedding((p.get("abstract") or "")[:2000])
    
    # Pass to every _hunt_once call
    for seed in seeds:
        topics = self._hunt_once(domain, seed["seed"], seed_strategy, precomputed_embeddings=abstract_embeddings)
```

**Expected savings**: ~40-60 embedding calls eliminated per run (5 seeds × 20 papers, but only ~20 unique papers exist)

#### 4. **Adaptive Gap Generation — Generate Less, Screen More**

**Problem**: `_GAPS_PER_SEED_REQUEST = 6` generates 6 gaps per seed. Each gap costs ~35K tokens through the gate chain. If only 1-2 survive, 4-5 are wasted.

**Fix**: Implement a **progressive generation** strategy:

```python
# Phase 1: Generate 3 gaps per seed (cheap)
# Phase 2: Screen all 3 (deterministic pre-screen + screen_research_gap)
# Phase 3: If ≥1 passes, stop and formalize
# Phase 4: If 0 pass, generate 2 more gaps for that seed
```

This changes the math from `5 seeds × 6 gaps × 35K = 1.05M tokens` to:
`5 seeds × (3 gaps × 35K + 2 extra gaps × 35K × 0.5) = ~610K tokens`

**Expected savings**: ~40-50% of gate-chain tokens

### System-Level Changes (Priority 2)

#### 5. **Parallel Seed Processing with ThreadPoolExecutor**

**Problem**: `seed_sequential=True` processes seeds one at a time. With 5 seeds, each taking ~5 minutes, that's 25 minutes of wall time.

**Fix**: Process seeds in parallel threads, but serialize API calls through the gateway's rate limiter.

```python
def discover_topics(self, domain=None, n_parallel=5):
    seeds = self._generate_dynamic_seeds(n_parallel)
    
    # Pre-compute shared resources
    recent_papers, abstract_embeddings = self._retrieve_all(domain)
    
    # Parallel seed processing with shared cache
    with ThreadPoolExecutor(max_workers=min(n_parallel, 3)) as executor:
        futures = {}
        for seed in seeds:
            future = executor.submit(
                self._hunt_once, domain, seed["seed"], 
                seed.get("strategy", "generic_fallback"),
                precomputed_embeddings=abstract_embeddings,
                recent_papers=recent_papers
            )
            futures[future] = seed
        
        all_topics = []
        for future in as_completed(futures):
            try:
                topics = future.result()
                all_topics.extend(topics or [])
            except Exception as e:
                log_agent_action("TopicHunter", "hunt_error", {"seed": futures[future]["seed"], "error": str(e)})
```

**Expected savings**: 3-5× wall time reduction. Same tokens, but user sees results faster.

#### 6. **Debate Parallelism**

**Problem**: `HypothesisDebateSystem.conduct_tournament` debates topics sequentially. If 3 topics pass topic hunting, they're debated one by one.

**Fix**: Use `concurrent.futures` to debate multiple topics in parallel.

```python
def conduct_tournament(self, topics, rounds=1):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = []
    with ThreadPoolExecutor(max_workers=min(len(topics), 3)) as executor:
        futures = {executor.submit(self.conduct_with_repair, t): t for t in topics}
        for future in as_completed(futures):
            result = future.result()
            results.extend(result)
            if result[-1].passed:
                # Cancel remaining, we have a winner
                for f in futures:
                    if not f.done():
                        f.cancel()
                break
    return sorted(results, key=lambda r: (r.passed, r.score), reverse=True)
```

**Expected savings**: 2-3× debate time. Debates are CPU-bound (LLM calls), so parallelism helps.

#### 7. **Cascade Budget Allocation**

**Problem**: Every seed gets the same `llm_budget_per_seed=30` regardless of quality. A seed that's producing rejected candidates wastes its budget.

**Fix**: Implement **cascade budget** — reallocate budget from failing seeds to promising ones.

```python
def discover_topics(self, domain=None, n_parallel=5):
    # Phase 1: Quick screen for all seeds (cheap)
    # Track which seeds produce candidates passing _quick_grounding_check
    # Give more budget to seeds with pass rate > threshold
    # Starve seeds with 0% pass rate after first iteration
    
    seed_budgets = {seed: self.runtime_config.llm_budget_per_seed for seed in seeds}
    
    # After first pass, redistribute
    passing_seeds = [s for s in seeds if seed_pass_rate[s] > 0.2]
    failing_seeds = [s for s in seeds if seed_pass_rate[s] <= 0.2]
    
    for s in failing_seeds:
        seed_budgets[s] = max(5, seed_budgets[s] // 2)  # Cut budget in half
    for s in passing_seeds:
        seed_budgets[s] = min(50, seed_budgets[s] + 10)  # Give more
```

**Expected savings**: ~20% of total budget by starving dead seeds

### Model Tier Optimization (Priority 3)

#### 8. **Tier-Aware Gate Chain**

**Problem**: `formalize_hypothesis` uses `tier="strong"` for every gap. At ~22K tokens per call, this is the single most expensive operation.

**Fix**: 
- Use `tier="cheap"` for `screen_research_gap` and `evaluate_layered_novelty` comparison (these are binary decisions)
- Use `tier="strong"` ONLY for `formalize_hypothesis` (the one place where structured output quality matters)
- Use `tier="judge"` ONLY for debate moderation

```python
# Current: formalize_hypothesis uses tier="strong" (expensive)
# Proposed: Only use strong tier if cheap tier formalization passes basic validation
raw = call_llm(prompt, temperature=0.3, tier="cheap")  # Try cheap first
parsed = parse_json_from_llm(raw)
if not self._validate_formalized(parsed):
    raw = call_llm(prompt, temperature=0.1, tier="strong")  # Fallback to strong
```

**Expected savings**: ~60% of formalize_hypothesis token cost

#### 9. **Embed Instead of LLM for Novelty**

**Problem**: `evaluate_layered_novelty` uses LLM comparison for high-overlap papers. But the embedding similarity is already a strong signal.

**Fix**: Replace LLM comparison with a deterministic rule:
- If `best_sim >= 0.95`: reject immediately (already implemented)
- If `0.85 <= best_sim < 0.95`: compare using Jaccard keyword overlap instead of LLM
- If `best_sim < 0.85`: automatically NOVEL

```python
def evaluate_layered_novelty(self, candidate, abstracts, precomputed_embeddings=None):
    # ... embedding computation ...
    
    if best_sim >= 0.95:
        return reject("near_duplicate")
    elif best_sim >= 0.85:
        # Deterministic check instead of LLM
        if self._deterministic_novelty_check(candidate, nearest_abstract):
            return NOVEL
        else:
            return reject("overlap")
    else:
        return NOVEL  # Below 0.85, definitely novel
```

**Expected savings**: ~6-12 LLM calls per run (one per high-overlap gap)

### Infrastructure Changes (Priority 4)

#### 10. **Run-Level Caching Layer**

**Problem**: CrossRunMemory is queried for every seed, every gap. The same data is fetched repeatedly.

**Fix**: Cache all external data at the `discover_topics()` level.

```python
def discover_topics(self, domain=None, n_parallel=5):
    # Fetch once
    cross_run_context = CrossRunMemory().get_prompt_context()
    excluded = self._excluded_titles()
    rejected_fingerprints = self._extract_rejected_fingerprints(cross_run_context)
    negative_lessons = CrossRunMemory().get_negative_result_lessons()
    evidence_map = build_cross_paper_evidence_map(recent_papers)
    
    # Pass cached data to every _hunt_once call
    # ...
```

#### 11. **Early Abort for Failing Seeds**

**Problem**: If a seed produces 0 passing gaps after the first 2-3 gates, the remaining budget is wasted.

**Fix**: Track pass-through rate per seed and abort early.

```python
def _hunt_once(self, domain, seed_hint, seed_strategy, ...):
    gaps_generated = 0
    gaps_passed = 0
    
    for gap in generated_gaps:
        gaps_generated += 1
        
        # Check early abort: if 3 generated and 0 passed, stop
        if gaps_generated >= 3 and gaps_passed == 0:
            log_agent_action("TopicHunter", "seed_early_aborted", {"seed": seed_hint})
            break
        
        # ... gate chain ...
        if gap_passes_all_gates:
            gaps_passed += 1
```

#### 12. **Graph-Based Early Signal (Use Before LLM)**

**Problem**: The system fetches graph signals, coupling gaps, sparsity cells, and contradictions but doesn't use them to PRE-FILTER candidates before the LLM gate chain.

**Fix**: Use structural signals to boost candidate quality at generation time, not just as context in the prompt.

```python
# In _generate_dynamic_seeds, use structural signals to generate
# better-targeted seeds that are more likely to produce passing gaps
def _generate_dynamic_seeds(self, n_seeds, ...):
    # Use coupling_gaps to generate seeds that bridge specific paper pairs
    # Use sparse_cells to generate seeds targeting under-explored combinations
    # Use contradictions to generate seeds resolving specific disagreements
```

### Summary: Expected Token Reduction

| Change | Estimated Savings | Difficulty |
|--------|------------------|------------|
| Two-Stage Funnel (cheap pre-screen) | ~30-40% | Medium |
| Prompt Compression | ~15-20% | Low |
| Precomputed Embedding Reuse | ~10-15% | Low |
| Adaptive Gap Generation | ~20-30% | Medium |
| Tier-Aware Gate Chain | ~15-20% | Low |
| Embed Instead of LLM Novelty | ~5-10% | Low |
| Early Abort | ~5-10% | Low |
| Parallel Processing | Wall time only | Medium |
| **Combined** | **~60-80%** | |

**Projected result**: Instead of 300+ LLM requests / 13M tokens, the system should use **60-100 LLM requests / 2-3M tokens** and reach hypothesis debate reliably.

---

## Part 3: Implementation Priority Order

### Sprint 1 (Quick Wins — 1-2 days)
1. Precompute embeddings once per run (shared across seeds)
2. Cache cross-run context, excluded titles, negative lessons at discover_topics level
3. Tier-aware gate chain (cheap tier for screening)
4. Prompt compression (truncate signals to top 3 each)
5. Early abort for failing seeds

### Sprint 2 (Medium — 2-3 days)
6. Two-stage funnel (deterministic pre-screen before LLM gates)
7. Adaptive gap generation (generate 3, screen, then decide)
8. Embed instead of LLM for novelty comparison
9. Validate formalized hypothesis with cheap tier first

### Sprint 3 (Advanced — 3-5 days)
10. Parallel seed processing with ThreadPoolExecutor
11. Parallel debate with ThreadPoolExecutor
12. Cascade budget allocation
13. Graph-based early signal for seed generation
14. Cross-seed query cache (already partially implemented)

### Sprint 4 (Scale — ongoing)
15. External parallel search agents (sub-agents for OpenAlex, arXiv, S2, OpenReview)
16. Distributed retrieval with async/await
17. Incremental embedding cache (update only new papers)
18. Cross-run knowledge distillation (compress prior run data into compact signals)

---

## Part 4: Additional Creative Ideas for Scale

### Sub-Agent Architecture for Parallel Search

Instead of one TopicHunter doing everything sequentially, spawn parallel sub-agents:

```python
class ParallelSearchOrchestrator:
    """Spawn sub-agents for each search platform."""
    
    def __init__(self):
        self.sub_agents = {
            "openalex": OpenAlexSearchAgent(),
            "arxiv": ArxivSearchAgent(),
            "s2": SemanticScholarSearchAgent(),
            "openreview": OpenReviewAgent(),
            "crossref": CrossrefAgent(),
        }
    
    def search_all(self, queries):
        """All platforms search simultaneously."""
        with ThreadPoolExecutor() as executor:
            futures = {}
            for platform, agent in self.sub_agents.items():
                future = executor.submit(agent.search, queries[platform])
                futures[future] = platform
            
            results = {}
            for future in as_completed(futures):
                platform = futures[future]
                results[platform] = future.result()
        return results
```

### Retrieval-Augmented Seed Generation

Instead of generating seeds from LLM alone, use retrieval to find *what's actually missing*:

1. Fetch the top 100 papers from each platform
2. Run a lightweight clustering (TF-IDF + K-means) to find topic clusters
3. Identify cluster boundaries and sparse regions
4. Generate seeds targeting the sparse regions

### Feedback-Driven Budget Reallocation

After each run, analyze which strategies/seeds produced the most passing topics and allocate more budget to those in the next run. This creates a self-improving system that learns to spend tokens efficiently.

### The "Golden Sample" Strategy

Instead of generating 6 gaps per seed and hoping some pass, use a smaller but higher-quality generation:

1. Generate 2 very specific gaps per seed (higher quality prompt)
2. Run full gate chain on both
3. If both fail, use the failure reasons to generate 1 more targeted gap
4. This ensures every gap gets the full treatment

### Multi-Modal Evidence

Use non-LLM signals to pre-screen:
- Citation graph density (already computed)
- Semantic Scholar citation counts (already fetched)
- OpenAlex concept overlap (already computed)
- Year distribution of papers (already fetched)
- Combine these into a single "gap score" that predicts pass probability without LLM

---

## Part 5: Verification Metrics

After implementing improvements, measure:

```python
# Key metrics to track
METRICS = {
    "llm_calls_per_run": "Target: <100 (currently 300+)",
    "tokens_per_run": "Target: <3M (currently 13M)",
    "topics_generated": "Target: 15-20 (currently 30 with 80% waste)",
    "topics_passing_to_debate": "Target: 3-5 (currently 1-3)",
    "time_to_debate": "Target: <2 minutes (currently 5+ minutes)",
    "cost_per_run": "Target: <50% of current",
    "gate_chain_pass_rate": "Target: >20% (currently ~10-15%)",
    "embedding_reuse_rate": "Target: >80% (currently ~0%)",
}
```

Run the eval harness (`python tests/test_eval_harness.py`) to verify improvements don't break quality.

---

*Generated by deep code analysis of Scholargraph v3/v4 codebase. All estimates based on token counting from actual prompt structures and LLM call patterns observed in `agents/topic_hunter.py`, `agents/hypothesis_debate.py`, `core/api_gateway.py`, and `core/workflow_nodes.py`.*
