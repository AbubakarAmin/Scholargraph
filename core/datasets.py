"""Curated local dataset catalog for deterministic planning."""

from __future__ import annotations

import json
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
    "openreview_calibration": {
        "name": "openreview_calibration",
        "version": "1",
        "source": "OpenReview API v2 (ICLR 2025-2026)",
        "access_policy": "local-only",
        "max_rows": 10000,
        "license": "CC-BY-4.0",
        "path": "data/review_calibration",
        "description": (
            "ICLR papers with reviewer scores, decisions, and meta-reviews "
            "for calibrating Supervisor.evaluate_section() and "
            "Editor.final_manuscript_referee() in Phase 4b."
        ),
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
    if name == "openreview_calibration":
        base = Path(spec.get("path", "data/review_calibration"))
        if not base.exists():
            return {"spec": spec, "rows": [], "row_count": 0, "manifest": {}}
        manifest_path = base / "_manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows: List[Dict[str, Any]] = []
        for venue_dir in sorted(base.iterdir()):
            if not venue_dir.is_dir() or venue_dir.name.startswith("_"):
                continue
            for paper_file in sorted(venue_dir.glob("*.json")):
                try:
                    paper = json.loads(paper_file.read_text(encoding="utf-8"))
                    rows.append(paper)
                except (json.JSONDecodeError, OSError):
                    continue
        return {"spec": spec, "rows": rows, "row_count": len(rows), "manifest": manifest}
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
