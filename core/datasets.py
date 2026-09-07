"""Curated local dataset catalog for deterministic planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import csv


DATASET_CATALOG: Dict[str, Dict[str, Any]] = {
    "bundled_synthetic": {
        "name": "bundled_synthetic",
        "version": "1",
        "source": "ScholarGraph generator",
        "access_policy": "local-only",
        "max_rows": 10000,
        "license": "generated",
        "path": None,
    },
    "sklearn_iris": {
        "name": "sklearn_iris",
        "version": "1",
        "source": "scikit-learn bundled dataset",
        "access_policy": "local-only",
        "max_rows": 150,
        "license": "BSD-3-Clause",
        "path": "data/catalog/iris.csv",
    },
    "sklearn_digits": {
        "name": "sklearn_digits",
        "version": "1",
        "source": "scikit-learn bundled dataset",
        "access_policy": "local-only",
        "max_rows": 1797,
        "license": "BSD-3-Clause",
        "path": None,
    },
}


def list_datasets() -> List[Dict[str, Any]]:
    return [dict(value) for value in DATASET_CATALOG.values()]


def resolve_dataset(name: str) -> Dict[str, Any]:
    """Resolve only catalogued, local datasets; never download implicitly."""
    if name not in DATASET_CATALOG:
        raise ValueError(f"Dataset is not in the local catalog: {name}")
    return dict(DATASET_CATALOG[name])


def load_local_dataset(name: str) -> Dict[str, Any]:
    """Load a catalogued dataset without network access."""
    spec = resolve_dataset(name)
    if spec.get("path"):
        path = Path(spec["path"])
        rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
        return {"spec": spec, "rows": rows, "row_count": len(rows)}
    if name == "sklearn_iris":
        from sklearn.datasets import load_iris

        dataset = load_iris(as_frame=True)
        return {"spec": spec, "rows": dataset.frame.to_dict(orient="records"), "row_count": len(dataset.frame)}
    if name == "sklearn_digits":
        from sklearn.datasets import load_digits

        dataset = load_digits(as_frame=True)
        return {"spec": spec, "rows": dataset.frame.to_dict(orient="records"), "row_count": len(dataset.frame)}
    return {"spec": spec, "rows": [], "row_count": 0}
