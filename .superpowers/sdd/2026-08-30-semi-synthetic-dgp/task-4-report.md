# Task 4 implementation report

## Changed files

- `src/skcausal/datasets/semi_synthetic_categorical.py` — adds the fixed-strength categorical assignment and Gaussian outcome DGP.
- `tests/datasets/test_semi_synthetic_categorical_dgp.py` — adds the four prescribed categorical behavior tests.
- `.superpowers/sdd/2026-08-30-semi-synthetic-dgp/task-4-report.md` — records implementation and verification evidence.

The shared `progress.md` ledger was not modified.

## Decisions

- Kept the constructor surface exactly at Task 4: no `target_confounding_bias` parameter and no Task 5 bias-calibration behavior.
- Stored string treatment levels `"0"` through `str(K - 1)` and relied on the base treatment coercion to enforce Polars categorical output.
- Calibrated the confounded softmax component to the requested source-population marginal, so mixing it with the same randomized target distribution preserves the exact marginal.
- Centered each treatment-only score's coefficients across treatment categories, preserving softmax identifiability.
- Used the first treatment level as the zero/reference outcome basis and applied main and modifier effects only to the remaining levels.
- Added a private categorical-class NumPy covariate normalization override. The base converts NumPy covariates to generically named Polars columns; restoring the frozen source names locally is required for the prescribed NumPy oracle behavior while named pandas/Polars frames retain strict schema checks.
- Kept all structural helpers private; the only concrete-class public additions are the required `get_levels` and `get_test_params` methods.

## TDD evidence

### RED

Command:

```text
uv run pytest tests/datasets/test_semi_synthetic_categorical_dgp.py -q
```

Result: collection failed as prescribed with `ModuleNotFoundError: No module named 'skcausal.datasets.semi_synthetic_categorical'` (exit 2).

### GREEN iterations

First implementation run, same command: 3 passed and 1 failed. The backend test exposed that NumPy covariates reached the frozen score map with generic `column_0` names instead of the source schema.

After the minimal categorical-class normalization fix, same command:

```text
4 passed in 0.94s
```

Compatibility command:

```text
uv run pytest tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py -q
```

Result: `20 passed in 1.21s`.

## Verification

Baseline before edits:

```text
uv run pytest -q
733 passed, 419 warnings in 5.82s
```

Formatting and lint:

```text
uv run ruff format src/skcausal/datasets/semi_synthetic_categorical.py tests/datasets/test_semi_synthetic_categorical_dgp.py
uv run ruff check src/skcausal/datasets/semi_synthetic_categorical.py tests/datasets/test_semi_synthetic_categorical_dgp.py
git diff --check
```

Final result: `2 files left unchanged`, `All checks passed!`, and `git diff --check` exited 0. An earlier lint pass correctly reported `RUF012` for the mutable class attribute; annotating `column_types` with `ClassVar` resolved it.

Focused plus dataset discovery/clone compatibility:

```text
uv run pytest tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_all_datasets.py -q
135 passed in 2.80s
```

Full repository suite after implementation:

```text
uv run pytest -q
743 passed, 419 warnings in 4.31s
```

## Review and concerns

- Local requirements/diff review found no Task 4 scope gaps and confirmed that Task 5 behavior is absent.
- The full suite retains 419 pre-existing deprecation/data-conversion warnings; Task 4 introduced no new warning category in the test output.
- `progress.md` remains untouched for the parent controller to update.
