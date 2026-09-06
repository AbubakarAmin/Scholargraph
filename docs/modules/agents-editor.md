# `agents/editor.py`

## Responsibility

Assembles the final paper, derives limitations from unresolved debate objections, resolves DOI/arXiv bibliography entries, generates LaTeX, and writes a companion experiment repository.

## Main API

- `create_final_paper()` builds the structured final paper.
- `generate_latex()` writes LaTeX and bibliography artifacts.
- `assemble_paper()` provides a combined compatibility API.

## Output

Generated files are placed under `output/` and the companion repository under the configured companion directory.
