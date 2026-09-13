# Sparsity Matrix — `core/sparsity_matrix.py`

Method × domain sparsity detection.

## Purpose

LLM-extracted (method, domain) pairs from retrieved abstracts are counted in a matrix. Rare combinations of well-established methods and domains are surfaced as sparse-cell gap signals — opportunities to apply a known method to an under-explored domain.

## How it works

1. Extract (method, domain) pairs from retrieved paper abstracts
2. Build a count matrix of method × domain
3. Identify sparse cells: methods with many applications but few in a specific domain
4. Inject these as gap signals into the TopicHunter discovery prompt

## Configuration

- `SPARSITY_MATRIX_ENABLED` (default: `true`): enable/disable

## Integration

Called by `TopicHunterAgent` during seed generation alongside structural gap mining and contradiction mining.
