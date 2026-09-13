# Structural Gaps — `core/structural_gaps.py`

Bibliographic coupling gap analysis via Semantic Scholar references.

## Purpose

Identifies pairs of papers that share references but do not cite each other (bibliographic coupling). These pairs are signals for structural research gaps: they work on related topics using overlapping foundations but have not engaged with each other's work.

## How it works

1. For retrieved papers, fetch their reference lists via Semantic Scholar
2. Build a paper × reference incidence matrix
3. Find pairs with high reference overlap but no mutual citation
4. Inject these as structural gap signals into the TopicHunter discovery prompt

## Configuration

- `STRUCTURAL_GAP_MINING_ENABLED` (default: `true`): enable/disable
- `STRUCTURAL_GAP_MAX_PAIRS` (default: `8`): maximum pairs to inject

## Integration

Called by `TopicHunterAgent` during seed generation. Results are injected as additional context into the discovery prompt, giving the LLM explicit structural signals to reason about.
