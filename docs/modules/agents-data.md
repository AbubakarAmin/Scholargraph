# DataAgent — `agents/data.py`

Dataset validation and provenance steward.

## Purpose

`DataAgent` validates user-provided datasets before they enter the experiment pipeline. It enforces schema, target, and integrity checks, and produces a `DatasetArtifact` with content hashing for downstream provenance.

## When it runs

In the `data_validation` phase, after planning and before writing/engineering. Only runs when the plan declares an explicit `dataset_path` or `dataset_file`.

## Checks performed

| Check | Behavior |
|---|---|
| File exists | Reject if path does not resolve |
| Parseable format | CSV or JSON; reject otherwise |
| Non-empty | Reject if zero rows |
| Column presence | Reject if declared features or target are missing |
| Target separation | Reject if target column overlaps feature columns |
| All-missing columns | Flag columns where every value is null |
| Schema snapshot | Record column names, dtypes, row count |
| Content hash | SHA-256 of the raw file for provenance |

## Output

Returns a `DatasetArtifact` (TypedDict in `core/contracts.py`) containing:
- `location`: absolute file path
- `content_hash`: SHA-256 digest
- `row_count`: number of rows
- `schema`: column → dtype mapping
- `validation`: `VerificationReport` with passed/score/note

## Invalid datasets

If validation fails, the run terminates with a visible terminal error. The system does not attempt to guess or repair the dataset.

## Capability manifest

```
Allowed: literature.search, literature.fetch, dataset.catalog, dataset.download, artifact.write
Forbidden: code.generate, code.execute, analysis.statistics, verification.claims
```
