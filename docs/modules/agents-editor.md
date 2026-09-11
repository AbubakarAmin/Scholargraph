## Module Overview

EditorAgent — submission-grade LaTeX, DOI bibliography, Limitations, companion repo.
Provides create_final_paper / generate_latex expected by main.py.

# `agents/editor.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Assembles the final paper, derives limitations from unresolved debate objections, resolves DOI/arXiv bibliography entries, generates LaTeX, and writes a companion experiment repository.

## Main API

- `create_final_paper()` builds the structured final paper.
- `generate_latex()` writes LaTeX and bibliography artifacts.
- `assemble_paper()` provides a combined compatibility API.

## Output

Generated files are placed under `output/` and the companion repository under the configured companion directory. The editor produces a reviewable paper with `publishable=false`; LaTeX export is deferred until the human approval endpoint is called. The companion manifest records seeds, contract hashes, Git commit, Python version, and pinned package versions.
