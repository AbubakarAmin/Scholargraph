# Datasets — `core/datasets.py`

Curated local dataset catalog for deterministic planning.

## Purpose

Provides a local-only dataset catalog that planners must choose from. This prevents the system from planning around live downloads that may fail, rate-limit, or be unreachable from the sandbox.

## Catalogued datasets

| Name | Source | Max rows | License | Access |
|---|---|---|---|---|
| `bundled_synthetic` | ScholarGraph generator | 10,000 | generated | local-only |
| `sklearn_iris` | scikit-learn | 150 | BSD-3-Clause | local-only |
| `sklearn_digits` | scikit-learn | 1,797 | BSD-3-Clause | local-only |
| `openreview_calibration` | OpenReview API v2 (ICLR 2025-2026) | 10,000 | CC-BY-4.0 | local-only |

## API

### `list_datasets()`
Returns a list of all catalogued dataset specs.

### `resolve_dataset(name)`
Resolves a dataset by name. Raises `ValueError` if not in the catalog.

### `load_local_dataset(name)`
Loads a catalogued dataset without network access. Returns `{"spec": ..., "rows": [...], "row_count": N, "manifest": {...}}`.

## Adding datasets

To add a dataset, add an entry to `DATASET_CATALOG` with:
- `name`: unique identifier
- `version`: version string
- `source`: data source description
- `access_policy`: always `"local-only"`
- `max_rows`: maximum row count
- `license`: license identifier
- `path`: relative path to the data file (or `None` for sklearn loaders)

## Capability integration

TopicHunter checks `dataset_plan` against the catalog before bridge validation. Uncatalogued datasets are rejected early with `reason_code="dataset_not_catalogued"`.
