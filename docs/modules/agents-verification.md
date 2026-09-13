# VerificationAgent — `agents/verification.py`

Artifact integrity and statistical mismatch detection.

## Purpose

`VerificationAgent` independently checks that execution artifacts are intact, that analysis reports exist, and that reported statistics agree with raw results. Findings can be blocking — the supervisor refuses the editing route when any blocking finding is present.

## When it runs

In the `independent_validation` phase, after execution and analysis.

## Checks performed

| Check | What it verifies |
|---|---|
| Raw result path exists | The file referenced by `raw_results_path` is on disk |
| Content hash matches | SHA-256 of the raw file matches the execution artifact's `content_hash` |
| Execution succeeded | The execution artifact's `status` is `success` |
| Analysis report exists | An independent analysis report is present for this experiment |
| Statistics agree | `core.verification.verify_statistics` confirms reported vs raw metrics |

## Output

Returns a list of `VerificationFinding` dicts, each with:
- `severity`: `blocking` or `advisory`
- `check`: which check produced the finding
- `message`: human-readable description
- `blocking`: boolean flag

## Blocking vs advisory

- **Blocking**: raw path missing, hash mismatch, execution failed, statistics disagree
- **Advisory**: missing analysis report (may be legitimate for some experiment types)

## Capability manifest

```
Allowed: artifact.read, verification.replay, verification.claims, artifact.write
Forbidden: code.generate, dataset.download, analysis.statistics
```
