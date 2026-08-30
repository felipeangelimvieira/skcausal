# Task 7 report — continuous oracle-bias calibration

## Scope and files

Implemented only Task 7's continuous target-bias calibration on base commit
`6f1e5f0120e889166b174446026a2ad871d6381d`.

Modified:

- `src/skcausal/datasets/semi_synthetic_continuous.py`
- `tests/datasets/test_semi_synthetic_continuous_dgp.py`
- `.superpowers/sdd/2026-08-30-semi-synthetic-dgp/task-7-report.md`

The ledger `progress.md` was read but not modified or staged. No public
diagnostics object or method was added, and SciPy was not added or used.

## Implementation decisions

- Added the optional constructor parameter `target_confounding_bias`, stored it
  before base initialization for clone compatibility, and rejected negative or
  nonfinite targets.
- Froze the prescribed 31-row intervention grid once during structural
  preparation and precomputed its source-population conditional outcome means.
  Candidate strength evaluation changes only the assignment density.
- Refactored the existing exact continuous density into the private
  `_log_prob_at_strength` path. Public `_log_prob` delegates to that path with
  `confounding_strength_`; therefore the fixed-strength transformed mixture,
  truncation-normalizer, support, and Jacobian behavior is unchanged.
- At each frozen intervention, candidate log densities are normalized with the
  existing `_normalized_log_weights` helper. The weighted observational response
  minus the unweighted causal response is computed row-wise, and raw bias is the
  RMS of those 31 discrepancies.
- Normalized bias divides the raw RMS by the frozen baseline-risk standard
  deviation. Both fixed-strength and calibrated construction store
  `confounding_strength_`, `confounding_bias_`, and
  `confounding_bias_ratio_`.
- Fully randomized assignment returns exact zero bias by construction. A
  positive target under full randomization raises an explicit unattainable-target
  error; a zero target selects zero fitted confounding strength.
- Added an independent fixed-strength test that reconstructs the 31-point RMS
  from the public `predict_y` and `log_prob` oracles, protecting both the frozen
  grid and the observational-versus-causal discrepancy definition.
- The requested review skill could not dispatch a reviewer because the task
  explicitly prohibited subagents. I instead performed a manual
  requirement-by-requirement diff review and ran the continuous exact-density,
  categorical, base, helper, dataset-contract, and full repository suites.

## TDD evidence

### RED cycle 1

Command:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py -q
```

Result before production changes: `6 failed, 18 passed in 2.00s`. The target
and invalid-target cases failed because `target_confounding_bias` was not an
accepted constructor argument; fixed-strength and randomized cases failed
because the fitted bias attributes did not exist. These were the expected
missing-feature failures.

### GREEN cycle 1

The same command after the minimal calibration implementation produced:
`24 passed in 4.48s`.

### RED/GREEN cycle 2

The fully randomized positive-target guard was separately removed before its
regression assertion was retained:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py::test_fully_randomized_continuous_assignment_has_zero_oracle_bias -q
```

RED result: `1 failed in 1.36s`, specifically `Failed: DID NOT RAISE
ValueError`. After restoring the minimal guard, GREEN results were `1 passed in
1.10s` for the isolated test and `24 passed in 4.24s` for the continuous file.

## Verification evidence

Formatting and lint:

```text
uv run ruff format src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
uv run ruff check src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
```

Result: `1 file reformatted, 1 file left unchanged`; `All checks passed!`.

Focused and compatibility suites:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_all_datasets.py -q
```

Result: `174 passed in 6.68s`.

Fresh full suite:

```text
uv run pytest -q
```

Result: `782 passed, 419 warnings in 9.76s`. The warning output contains the
repository's existing pandas/sklearn/Polars deprecations and Optuna user
warnings; Task 7 produced no failures.

Final staged-tree pre-commit gate:

```text
uv run ruff format --check src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
uv run ruff check src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_all_datasets.py -q
uv run pytest -q
```

Result: `2 files already formatted`; `All checks passed!`; `174 passed in
6.64s`; full suite `782 passed, 419 warnings in 9.59s`.

## Concerns and deferred work

- No correctness concern remains within Task 7 scope.
- Public exports and discovery remain Task 8 scope; this task changes only the
  concrete continuous constructor and its prescribed fitted attributes.
- The generic calibration helper's bracketing and tolerance policy is retained
  unchanged from Task 3.
