# Final whole-branch fix report

Base SHA: `aadced800152923791411d1afd646785c9471eef`

Commit message: `fix: harden semi-synthetic numerical contracts`

## Files changed

- `docs/superpowers/specs/2026-08-30-semi-synthetic-dgp-design.md`
- `docs/superpowers/plans/2026-08-30-semi-synthetic-dgp.md`
- `src/skcausal/datasets/semi_synthetic.py`
- `src/skcausal/datasets/semi_synthetic_categorical.py`
- `src/skcausal/datasets/semi_synthetic_continuous.py`
- `tests/datasets/test_semi_synthetic_base.py`
- `tests/datasets/test_semi_synthetic_helpers.py`
- `tests/datasets/test_semi_synthetic_categorical_dgp.py`
- `tests/datasets/test_semi_synthetic_continuous_dgp.py`
- `.superpowers/sdd/2026-08-30-semi-synthetic-dgp/final-fix-report.md`

`progress.md` was read as the governing ledger and was not modified or staged.

## Product-decision documentation

The design, implementation plan, and public
`ContinuousSemiSyntheticDataset` class documentation now state the exact
floating-point law:

- `treatment_domain="real"` retains the unconditioned latent Gaussian mixture;
- positive and bounded domains condition each Gaussian mixture component on
  the finite latent interval that round-trips to finite, distinct,
  representable interior treatments;
- sampling and `log_prob` use the same component-wise conditioned law,
  including truncated-Normal normalizers and the exact inverse Jacobian; and
- treatments are not clipped and the implementation does not claim the
  untruncated ideal-real transform law for transformed domains.

The pre-existing focused transformed-density regressions continue to verify
the truncated mixture and Jacobian contract.

## Fixes

- Categorical scalar strength, target, mixture-weight, effect, and noise
  settings now reject NaN and both infinities at constructor configuration
  boundaries with parameter-specific messages.
- Categorical inverse-CDF sampling constructs the CDF separately and forces
  its final entry to `1.0` before indexing.
- Bounded continuous domains require `lower < upper`, at least two distinct
  interior treatment floats, strictly ordered derived latent bounds, and an
  exact endpoint round-trip to distinct interior treatments.
- Normalized log weights reject empty/all-nonfinite input and NaN or positive
  infinity rather than returning NaN weights.
- Continuous preparation, calibration, treatment-basis, oracle-mean, and
  sampling paths reject non-representable derived values. Stored confounding
  strength/bias attributes are checked for finiteness.
- The three mutable test `column_types` attributes are annotated with
  `ClassVar`.

## TDD evidence

### RED

Command:

```text
uv run pytest tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_continuous_dgp.py -q
```

Result: exit 1, `18 failed, 56 passed, 101 warnings in 7.54s`. The failures
were the intended missing behaviors: all-nonfinite weight rejection,
parameter-specific categorical nonfinite validation, CDF undershoot closure,
the one-interior-float interval, the extreme finite active-confounding case,
and each requested continuous derived-state guard.

Command:

```text
uv run ruff check src/skcausal/datasets/semi_synthetic.py src/skcausal/datasets/semi_synthetic_categorical.py src/skcausal/datasets/semi_synthetic_continuous.py src/skcausal/datasets/__init__.py tests/datasets/__init__.py tests/datasets/_semi_synthetic_test_utils.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_continuous_dgp.py tests/test_lookup.py
```

Result: exit 1 with exactly three `RUF012` findings at
`test_semi_synthetic_base.py:11`, `:38`, and `:109`.

### GREEN

Command:

```text
uv run pytest tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_continuous_dgp.py tests/datasets/test_semi_synthetic_base.py -q
```

Result: exit 0, `80 passed in 4.55s`.

## Required verification

Command:

```text
uv run ruff check src/skcausal/datasets/semi_synthetic.py src/skcausal/datasets/semi_synthetic_categorical.py src/skcausal/datasets/semi_synthetic_continuous.py src/skcausal/datasets/__init__.py tests/datasets/__init__.py tests/datasets/_semi_synthetic_test_utils.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_continuous_dgp.py tests/test_lookup.py
```

Result: exit 0, `All checks passed!`.

Command:

```text
uv run ruff format --check src/skcausal/datasets/semi_synthetic.py src/skcausal/datasets/semi_synthetic_categorical.py src/skcausal/datasets/semi_synthetic_continuous.py src/skcausal/datasets/__init__.py tests/datasets/__init__.py tests/datasets/_semi_synthetic_test_utils.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_continuous_dgp.py tests/test_lookup.py
```

Result: exit 0, `11 files already formatted`.

Command:

```text
uv run pytest tests/datasets/test_semi_synthetic_regressor.py tests/datasets/test_semi_synthetic_classifier.py -q
```

Result: exit 0, `3 passed in 2.21s`.

Command:

```text
uv run pytest tests/datasets tests/datatypes -q
```

Result: exit 0, `259 passed, 10 warnings in 6.48s`.

Command:

```text
uv run pytest -q
```

Result: exit 0, `815 passed, 419 warnings in 9.44s`.

Command:

```text
git diff --check
```

Result: exit 0 with no output.

Command after the initial commit (`2992f53`), before this evidence-only amend:

```text
git status --short
```

Result: exit 0 with no output (clean worktree). The evidence-only amend
preserves one focused commit in history; final status is checked again during
handoff.

## Concerns

No unresolved implementation concern. The full-suite warnings are the same 419
pre-existing pandas/sklearn/Polars/Optuna warnings present at the clean baseline;
the focused semi-synthetic run emits no warning.
