## Module Overview

Curated local dataset catalog for deterministic planning.

# `core/datasets.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

The dataset catalog is local-only and versioned. It contains generated synthetic data, a checked-in Iris CSV asset, and loaders for scikit-learn bundled datasets. Dataset resolution never downloads data implicitly. Planner rescoping records the original infeasibility and the replacement catalog choice.

Key APIs: `list_datasets`, `resolve_dataset`, and `load_local_dataset`.
