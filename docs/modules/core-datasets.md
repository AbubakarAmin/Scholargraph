# `core/datasets.py`

The dataset catalog is local-only and versioned. It contains generated synthetic data, a checked-in Iris CSV asset, and loaders for scikit-learn bundled datasets. Dataset resolution never downloads data implicitly. Planner rescoping records the original infeasibility and the replacement catalog choice.

Key APIs: `list_datasets`, `resolve_dataset`, and `load_local_dataset`.