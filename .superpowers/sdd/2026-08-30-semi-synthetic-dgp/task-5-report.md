# Task 5 report — categorical oracle-bias calibration

## Changed files

- `src/skcausal/datasets/semi_synthetic_categorical.py`
  - Added the optional `target_confounding_bias` constructor parameter and
    nonnegative/finite validation.
  - Froze category-specific source outcome means after drawing the outcome
    surface.
  - Added private candidate-strength probability and bias evaluation.
  - Calibrated `confounding_strength_` with candidate-specific softmax
    intercepts while preserving the requested treatment marginals.
  - Added fitted `confounding_bias_` and `confounding_bias_ratio_` attributes.
  - Preserved configured fixed-strength behavior when no target is supplied;
    fully randomized assignment records exact zero bias and rejects positive
    targets as unattainable.
- `tests/datasets/test_semi_synthetic_categorical_dgp.py`
  - Added target-calibration, exact randomized-zero, invalid-target, and
    unreachable-target coverage.
- `.superpowers/sdd/2026-08-30-semi-synthetic-dgp/task-5-report.md`
  - Recorded implementation and verification evidence.

## Decisions

- The source outcome matrix is private (`_source_category_outcome_means`) so the
  only new public fitted surface is the prescribed strength/bias attributes; no
  diagnostics object or API was introduced.
- Candidate calibration recomputes softmax intercepts without mutating final
  state. The selected candidate's intercepts are stored only after strength
  calibration completes.
- Raw categorical bias is the range of the per-category observational-minus-
  causal distortions. The fitted ratio divides that range by the frozen source
  baseline standard deviation.
- `randomized_weight == 1.0` returns exact zero by construction. A positive
  target in that configuration raises `ValueError` with a fully randomized /
  unattainable explanation.
- The fixed-strength path retains the configured `confounding_strength` and the
  same calibrated assignment law used before Task 5.

## TDD evidence

### Baseline

- `uv run pytest tests/datasets/test_semi_synthetic_categorical_dgp.py -q`
  - First sandboxed attempt: environment error opening the shared uv cache.
  - Approved rerun: `4 passed in 2.48s`.

### RED

- Added all four behavior/validation tests before production changes.
- `uv run pytest tests/datasets/test_semi_synthetic_categorical_dgp.py -q`
  - Result: `4 failed, 4 passed in 1.03s`.
  - Expected failures: unsupported `target_confounding_bias`, missing
    `confounding_bias_ratio_`, and the two missing validation paths.

### GREEN

- `uv run pytest tests/datasets/test_semi_synthetic_categorical_dgp.py -q`
  - Result: `8 passed in 2.44s`.

### Independent formula check

- `uv run python -c '<public-oracle categorical bias calculation>'`
  - Fitted strength: `0.29561853408813477`.
  - Stored and independently calculated normalized bias:
    `0.3500003406048962` and `0.3500003406048963`.
  - Marginals: `[0.2999999999901173, 0.40000000001672187,
    0.29999999999316096]`.

## Final verification

- `uv run ruff format src/skcausal/datasets/semi_synthetic_categorical.py tests/datasets/test_semi_synthetic_categorical_dgp.py`
  - Result: `2 files reformatted`.
- `uv run ruff check src/skcausal/datasets/semi_synthetic_categorical.py tests/datasets/test_semi_synthetic_categorical_dgp.py`
  - Result: `All checks passed!`.
- `uv run pytest tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_regressor.py tests/datasets/test_semi_synthetic_classifier.py -q`
  - Result: `27 passed in 2.39s`.
- `uv run pytest -q`
  - Result: `747 passed, 419 warnings in 4.41s`.

## Concerns

- No Task 5 correctness concerns remain.
- The full suite's 419 warnings are pre-existing deprecation/data-conversion and
  Optuna parameter warnings; Task 5 adds no warning category.
