## Module Overview

Filesystem artifact serialization for completed research runs.

# `core/artifacts.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Serializes completed workflow state into the generated LaTeX file, YAML plan, and JSON run summary. It also provides `FilesystemArtifactStore`, a traversal-safe implementation of `ArtifactPort` for named runtime artifacts.

## Compatibility

`main.save_results()` remains as a thin wrapper for existing CLI, web, and test callers.
