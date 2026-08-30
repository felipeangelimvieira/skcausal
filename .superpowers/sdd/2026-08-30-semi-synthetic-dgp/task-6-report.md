# Task 6 report — scalar-continuous semi-synthetic DGP

## Scope and files

Implemented only Task 6's fixed-confounding-strength continuous DGP.
`target_confounding_bias`, continuous oracle-bias evaluation, and calibration are
intentionally deferred to Task 7. Public package exports remain Task 8 scope.

Created:

- `src/skcausal/datasets/semi_synthetic_continuous.py`
- `tests/datasets/test_semi_synthetic_continuous_dgp.py`
- `.superpowers/sdd/2026-08-30-semi-synthetic-dgp/task-6-report.md`

The ledger `progress.md` was read but not modified or staged.

## Implementation decisions

- Added one public concrete class, `ContinuousSemiSyntheticDataset`; all
  transformation, support, latent assignment, and basis operations remain
  private helpers.
- Preserved source `X` through the existing `BaseSemiSyntheticDataset`
  lifecycle. Replications reuse the frozen score map, baseline, coefficients,
  latent region, treatment-basis standardization, and hinge knots.
- Used separate confounded and randomized latent Gaussian components and the
  existing stable `_log_mixture` helper. Positive and bounded log-densities add
  the exact inverse-transformation Jacobian; invalid support values return
  negative infinity.
- Used exact real, exponential, and logistic-domain transforms, with saturation
  to the nearest representable open-support Float64 value only when the
  mathematical result overflows, underflows, or rounds to a closed bounded
  endpoint. The bounded forward transform uses a sign-split logistic calculation
  and overflow-safe convex interpolation.
- Froze the intervention region at the 1% and 99% quantiles of the randomized
  latent Gaussian. Froze linear, quadratic, sinusoidal, and two hinge basis
  terms on a deterministic 1001-point latent grid; hinge knots are the latent
  Gaussian 1/3 and 2/3 quantiles. The standardized basis at latent zero is
  subtracted so zero is the reference treatment.
- Added `get_test_params` with real and bounded configurations backed by
  `KangSchaferContinuous(n=64, random_state=4)`.
- Added focused behavior tests beyond the plan's minimum for an independently
  reconstructed bounded mixture/Jacobian density, exact source-X preservation,
  frozen structure across resampling, and NumPy/pandas/Polars oracle parity.
- The requested code-review skill could not dispatch a reviewer because this
  task explicitly prohibited subagents. A manual requirement-by-requirement
  diff review and the repository's dataset contract suite were used instead.

## TDD evidence

### RED

Command:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py -q
```

Result before production code existed: collection failed as intended with
`ModuleNotFoundError: No module named 'skcausal.datasets.semi_synthetic_continuous'`
(`1 error`). This was the expected missing-feature failure, not a test typo.

### GREEN

After the minimal continuous module was added:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py -q
```

Result: `7 passed in 2.74s`.

Prescribed focused compatibility command:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py -q
```

Result: `23 passed in 1.19s`.

## Verification evidence

Baseline before edits:

```text
uv run pytest -q
```

Result: `747 passed, 419 warnings in 5.86s`.

Formatting and lint:

```text
uv run ruff format src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
uv run ruff check src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
```

Result: `2 files reformatted`; `All checks passed!`.

Dataset discovery, cloning, backend, random-state, and dtype contracts:

```text
uv run pytest tests/datasets/test_all_datasets.py -q
```

Result: `126 passed in 5.50s`.

Categorical and legacy semi-synthetic compatibility:

```text
uv run pytest tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_regressor.py tests/datasets/test_semi_synthetic_classifier.py tests/datasets/test_semi_synthetic_continuous_dgp.py -q
```

Result: `18 passed in 3.72s`.

Fresh full suite on the final formatted implementation:

```text
uv run pytest -q
```

Result: `765 passed, 419 warnings in 12.69s`. The warnings are the same known
deprecation/data-conversion/user-warning classes present in the baseline; no
new failure or warning class was observed from Task 6.

Final pre-commit gate, after staging the three scoped files:

```text
uv run ruff format --check src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
uv run ruff check src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py -q
uv run pytest -q
```

Result: `2 files already formatted`; `All checks passed!`; `23 passed in
2.80s`; `765 passed, 419 warnings in 5.26s`.

## Concerns and deferred work

- No correctness concern remains within Task 6's fixed-strength scope.
- Continuous bias attributes and target calibration are deliberately absent;
  Task 7 will build them using the frozen source state and `_latent_mean(...,
  strength=...)` hook added here.
- `ContinuousSemiSyntheticDataset` is importable from its concrete module but
  is not exported from `skcausal.datasets`; Task 8 owns that integration.

## Fix round 1 — floating-point support boundaries

### Review findings verified

1. For bounded domain `(-2.0, 3.0)`, both
   `np.nextafter(-2.0, np.inf)` and `np.nextafter(3.0, -np.inf)` were valid
   treatment values but normalized to unit coordinates exactly equal to `0.0`
   and `1.0`. The inverse and Jacobian produced opposing infinities, their sum
   became NaN, and public `log_prob` raised `ValueError`.
2. Finite latent values could round the bounded logistic transform to exact
   endpoints, while the positive exponential transform could underflow to zero
   or overflow to infinity. This made construction fail for valid large noise
   or confounding-strength parameters.

### RED evidence

Bounded density at the two adjacent in-support Float64 values:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py::test_bounded_log_density_is_exact_at_representable_support_edges -q
```

Result before the inverse/Jacobian fix: `1 failed, 3 warnings in 2.40s`.
The failure was the public `_log_prob must return one non-NaN value per row`
guard after divide-by-zero and invalid-add warnings.

Extreme bounded and positive-domain construction/grid cases:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py::test_extreme_finite_parameters_keep_samples_and_grid_in_representable_support -q
```

Result before the forward-transform fix: `4 failed, 2 warnings in 1.39s`.
Failures covered bounded noise `100.0`, an overflow-width bounded domain,
positive noise `1000.0`, and positive confounding strength `10000.0`.

### Fix decisions

- Replaced bounded unit-coordinate inversion with direct log distances:
  `log(a - lower) - log(upper - a)`.
- Replaced the bounded inverse-Jacobian calculation with
  `log(upper - lower) - log(a - lower) - log(upper - a)`.
- Added a private positive-difference log helper that uses direct subtraction
  whenever representable and switches to `logaddexp` only when a cross-zero
  subtraction overflows. This keeps widths such as `(-float_max, float_max)`
  finite in log space.
- Replaced width multiplication in the bounded forward transform with convex
  interpolation and constrained only rounded endpoints to
  `nextafter(lower, upper)` / `nextafter(upper, lower)`.
- Constrained positive overflow/underflow only to the largest finite Float64 and
  the smallest positive Float64. Ordinary representable transform values are
  unchanged.
- Extended the extreme-parameter test to require finite exact log-density
  evaluation for generated treatments and all grid endpoints, in addition to
  open-support membership.

### GREEN and verification evidence

Isolated regressions after each fix:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py::test_bounded_log_density_is_exact_at_representable_support_edges -q
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py::test_extreme_finite_parameters_keep_samples_and_grid_in_representable_support -q
```

Results: `1 passed in 0.95s`; then `4 passed in 2.37s`. After adding density
coverage to the extreme test, it remained green with `4 passed in 1.15s`.

Formatting and lint:

```text
uv run ruff format src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
uv run ruff check src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
```

Result: `1 file reformatted, 1 file left unchanged`; `All checks passed!`.

Focused continuous/base/helper compatibility:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py -q
```

Result: `28 passed in 1.11s`.

Dataset discovery, categorical, and legacy semi-synthetic compatibility:

```text
uv run pytest tests/datasets/test_all_datasets.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_regressor.py tests/datasets/test_semi_synthetic_classifier.py -q
```

Result: `137 passed in 1.52s`.

Fresh full suite:

```text
uv run pytest -q
```

Result: `770 passed, 419 warnings in 4.70s`; warning classes match the existing
baseline.
