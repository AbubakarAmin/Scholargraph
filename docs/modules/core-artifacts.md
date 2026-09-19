# Artifacts

**File:** `core/artifacts.py` (154 lines)

## Purpose

Filesystem-based artifact serialization for completed or failed research runs — writes the final paper (LaTeX), research plan, QA answer, run summary, and failure dossiers.

## Key classes

| Class/Function | Purpose |
|---|---|
| `FilesystemArtifactStore(ArtifactPort)` | Stores named artifacts under a root directory with path traversal protection. Handles `bytes`, `str`, and JSON-serializable objects. |
| `save_results(state, output_dir)` | Main entry point for persisting run outputs: LaTeX paper (if human-approved), plan (YAML), QA answer (JSON), and `research_summary.json` |
| `save_failure_dossier(state, output_dir)` | Writes `failure_dossier.json` containing terminal error, evidence gate status, experiment outputs, verification findings, and the run summary |

## Behavior

- LaTeX output is only saved if `human_approved` is `True`.
- QA mode saves a `qa_answer.json` with query, answer, key findings, bibliography, and citation verification.
- On terminal error in QA mode, if a `qa_answer` exists, the paper is still saved (only failure dossier if no answer).
- `save_results` catches all exceptions and prints a warning — it never raises.

## Gotchas

- `ArtifactStore.save` validates that the resolved path stays inside the configured root directory (prevents path traversal).
- Failed runs produce a failure dossier, not a manuscript or companion repository.
- Artifacts are also recorded in the research DB for durable querying.
