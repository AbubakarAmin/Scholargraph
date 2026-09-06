"""Dataset stewardship agent: inspect and validate data without running experiments."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from core.capabilities import DEFAULT_MANIFESTS
from core.contracts import DatasetArtifact, DatasetSpec
from core.context import RunContext, get_active_context


class DataAgent:
    """Create immutable dataset metadata for downstream experiment workers."""

    capability_manifest = DEFAULT_MANIFESTS["DataAgent"]

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()

    def validate_dataset(self, path: str, spec: DatasetSpec) -> DatasetArtifact:
        dataset_path = Path(path).resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

        frame = self._read_frame(dataset_path)
        warnings = []
        target = spec.get("target")
        if target and target not in frame.columns:
            warnings.append(f"target column missing: {target}")
        feature_names = set(spec.get("features") or [])
        overlap = sorted(feature_names & {target} if target else set())
        if overlap:
            warnings.append(f"target appears in features: {overlap[0]}")
        if frame.empty:
            warnings.append("dataset has no rows")
        if frame.isna().all(axis=None):
            warnings.append("dataset contains only missing values")

        digest = self._sha256(dataset_path)
        now = datetime.now(timezone.utc).isoformat()
        passed = not warnings
        return {
            "location": str(dataset_path),
            "content_hash": digest,
            "row_count": int(len(frame)),
            "schema": {
                "columns": list(frame.columns),
                "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
            },
            "spec": spec,
            "validation": {
                "passed": passed,
                "score": 10.0 if passed else 0.0,
                "note": "Dataset validation completed",
                "mismatches": warnings,
                "checks": {
                    "exists": True,
                    "non_empty": not frame.empty,
                    "target_present": not target or target in frame.columns,
                    "target_not_in_features": not overlap,
                },
            },
            "provenance": {
                "artifact_type": "dataset",
                "source": str(dataset_path),
                "content_hash": digest,
                "created_at": now,
                "status": "validated" if passed else "needs_review",
                "limitations": warnings,
            },
        }

    @staticmethod
    def _read_frame(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".json":
            return pd.read_json(path)
        raise ValueError("Supported dataset formats are .csv and .json")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
