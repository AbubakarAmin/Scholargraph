# Contradiction Mining — `core/contradiction_mining.py`

Opposing-claim detection across papers.

## Purpose

LLM-assisted identification of papers making opposing empirical claims on the same subject. Contradictions are ready-made research gaps: they indicate unresolved scientific questions where controlled comparison could settle the dispute.

## How it works

1. Group retrieved papers by topical overlap
2. Use an LLM to identify pairs with opposing empirical claims
3. Extract the specific claim, evidence, and context from each side
4. Inject these as contradiction-based gap signals into the TopicHunter discovery prompt

## Configuration

- `CONTRADICTION_MINING_ENABLED` (default: `true`): enable/disable

## Integration

Called by `TopicHunterAgent` during seed generation. Works alongside structural gap mining and sparsity matrix analysis to provide multi-signal gap discovery.
