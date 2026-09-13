# Evidence Synthesis — `core/evidence_synthesis.py`

Cross-paper evidence map and bridge validation.

## Purpose

Builds a provenance-preserving corpus map from retrieved papers and proposes conservative bridges between papers with complementary method and application signals. A bridge is evidence for a candidate topic to screen, never proof that a research gap exists.

## Key functions

### `build_cross_paper_evidence_map(papers, *, max_papers=20, max_bridges=12)`

Builds a deterministic evidence graph:
1. Extracts salient terms from each paper's title + abstract
2. Classifies terms into roles: `method`, `evaluation`, `setting`
3. Finds directional transfer bridges: one paper contributes a method/evaluation, another contributes a distinct setting
4. Each bridge carries paper IDs and verbatim excerpts for auditor inspection

Returns a dict with `nodes` (paper list) and `bridges` (validated connections).

### `validate_candidate_bridge_claim(candidate, evidence_map)`

Verifies that a candidate topic's bridge claims are grounded in the evidence map:
- Checks that cited bridge IDs exist
- Verifies method/setting signals match
- Returns `{"valid": bool, "errors": [...]}`

Malformed or missing bridge IDs produce soft warnings (not hard rejects) with a programmatic fallback match.

### `validate_topic_admission(hypothesis)`

Validates that a structured hypothesis passes admission checks before debate:
- Requires a measurable research question
- Requires a falsification condition
- Requires a minimum viable experiment (dataset, models, metrics, baseline, seeds ≥ 3)
- Requires at least one baselines entry
- Returns `{"admitted": bool, "errors": [...]}`

## Bridge structure

Each bridge contains:
- `source_paper_id`: paper contributing the method/evaluation
- `target_paper_id`: paper contributing the setting
- `method_terms`: extracted method/evaluation terms
- `setting_terms`: extracted setting terms
- `verbatim_excerpt`: exact source text around the bridge signal
- `bridge_type`: `method_to_setting` or `evaluation_to_setting`
