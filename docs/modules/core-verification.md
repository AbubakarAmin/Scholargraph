# `core/verification.py`

## Responsibility

Deterministic evidence checks for citations, statistics, numeric result claims, and reproducibility dossiers.

## Key API

`extract_citation_ids`, `resolve_doi`, `resolve_arxiv`, `verify_citations`, `verify_statistics`, `preregister_power`, `validate_reviewer_checklist`, `final_manuscript_referee`, `hard_verify_section`, and `reproducibility_dossier`.

## Design rule

These checks are the hard quality gate. Supervisor LLM review runs after them and must not override a hard failure.

The final referee also requires uncertainty or sample-size context for quantitative claims, a non-empty limitations section, declared outcomes, and literature evidence for paper-specific citations.
