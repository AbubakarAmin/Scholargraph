## Module Overview

Hard verification checks: citation resolution + statistical validity.
These are deterministic — not LLM vibe scores.

# `core/verification.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Deterministic evidence checks for citations, statistics, numeric result claims, and reproducibility dossiers.

## Key API

- `extract_citation_ids`: DOI regex + arXiv id + author-year.
- `resolve_doi`: CrossRef `GET /works/{doi}`.
- `resolve_arxiv`: arXiv Atom API.
- `verify_citations`: Resolves DOIs/arXiv IDs and compares title/author metadata. Unresolved or mismatched → hard failure.
- `verify_statistics`: Re-derive mean/std (optional Welch p) from raw JSON; rtol default 0.05.
- `classify_claim`: Claims classified as `literature_reference`, `method_definition`, `planned_test`, or `empirical_result` before ledger entry.
- `cross_section_numeric_consistency`: Collects every numeric claim tied to the same experiment across all sections and diffs them. Conflicts are hard failures.
- `consistency_referee`: One isolated model pass over the full draft to find contradictions.
- `novelty_overlap_check`: Deterministic overlap-coefficient screen of Abstract+Introduction vs closest prior work / literature evidence.
- `preregister_power`: Computes sample-size requirements before execution.
- `validate_reviewer_checklist`: Enforces limitations, baselines, outcomes, uncertainty reporting, and literature evidence requirements.
- `final_manuscript_referee(sections, plan, outputs, topic=...)`: Full manuscript check including novelty gate.
- `hard_verify_section`: Citations + stats bundle for Supervisor.
- `reproducibility_dossier`: Deterministic artifact checklist.
- `PROHIBITED_MANUSCRIPT_TEXT`: Blocks leaked harness diagnostics from narrative sections.

## v4 upgrade (2026-09)

- **Novelty-plagiarism gate** (`novelty_overlap_check`): Deterministic overlap-coefficient screen wired into `final_manuscript_referee`.
- **Claim typing** (`classify_claim`): Only `empirical_result` claims tied to artifact IDs are eligible for verification.
- **Cross-section consistency** (`cross_section_numeric_consistency`): Hard failure when two sections cite different numbers for the same experiment.

## Design rule

These checks are the hard quality gate. Supervisor LLM review runs after them and must not override a hard failure.
