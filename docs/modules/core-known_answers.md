# Known Answers — `core/known_answers.py`

Known-answer validation fixtures for experiment code.

## Purpose

Before trusting generated experiment code on real data, run it against a synthetic case with a known closed-form or independently-verifiable answer. This catches silently-wrong implementations that would otherwise produce plausible-looking wrong numbers.

## How it works

1. For supported experiment types, a known-answer fixture is automatically generated
2. The fixture defines a synthetic input with a predictable output
3. The experiment code is executed against the fixture
4. Output metrics are compared against expected values within a tolerance
5. If the known-answer check fails, the code is sent back for repair

## Supported types

- Linear DMD on a linear system (should recover exact eigenvalues)
- Sinkhorn on identical source/target distributions (should return near-identity transport)
- Domain-specific algorithms with plan-provided fixtures

## Integration

Called by `EngineerAgent` during the branch search phase, before promoting a variant to full multi-seed execution. Also available via `core.sandbox.run_known_answer_check(code, expected_metrics, tolerance)`.
