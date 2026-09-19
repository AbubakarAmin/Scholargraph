## Module Overview

EditorAgent — submission-grade LaTeX, DOI bibliography, Limitations, companion repo.
Provides create_final_paper / generate_latex expected by main.py.

# `agents/editor.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Assembles the final paper, derives limitations from unresolved debate objections, resolves DOI/arXiv bibliography entries, generates LaTeX, and writes a companion experiment repository.

## Main API

- `create_final_paper()` builds the structured final paper. Checks run status first — failed runs produce a failure dossier, not a manuscript.
- `generate_latex()` writes LaTeX and bibliography artifacts.
- `assemble_paper()` provides a combined compatibility API.

## v4 upgrade (2026-09)

- **Editor referee repair** (`editor_repair_route`): Release-referee failures route to one bounded repair pass instead of terminal failure. Failed-experiment failures stay terminal. The graph edge from `editing` routes to `writing_results` when `editor_repair_count >= 1`.

## Output

Generated files are placed under `output/` and the companion repository under the configured companion directory. The editor produces a reviewable paper with `publishable=false`; LaTeX export is deferred until the human approval endpoint is called. The companion manifest records seeds, contract hashes, Git commit, Python version, and pinned package versions.

## Release gating

- Deterministic citation, numeric, dataset, checklist, reproducibility, and failure checks run before assembly.
- Human approval required via `POST /api/release/approve`.
- Failed runs are never exportable as manuscripts or companion repositories.
