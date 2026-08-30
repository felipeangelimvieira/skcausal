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
- Used exact real, exponential, and logistic-domain transforms without output
  clipping. Positive and bounded assignment components are Gaussian laws
  conditioned on a frozen, numerically invertible latent interval; sampling and
  component densities use the same truncation law and normalizer. The bounded
  forward transform uses a sign-split logistic calculation and overflow-safe
  convex interpolation.
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

## Fix round 2 — preserve the exact continuous assignment law

Round 1's output-saturation approach was superseded because it made the forward
maps many-to-one and introduced unreported point masses. The stable direct
bounded inverse/Jacobian calculations from round 1 remain valid and are retained.

### RED evidence

Round-trip and exact component-law tests were added first:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py::test_transforms_round_trip_on_restricted_invertible_latent_support tests/datasets/test_semi_synthetic_continuous_dgp.py::test_positive_log_density_matches_exact_truncated_latent_mixture -q
```

Result against the clipping implementation: `3 failed in 2.48s`. Both domains
lacked an invertible latent-support contract, and the positive mixture lacked
component-specific truncation normalizers.

An additional bounded likelihood-support test was added during diff review:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py::test_log_density_is_negative_infinity_outside_restricted_latent_support -q
```

Initial result: `1 failed in 1.25s`; the implementation extrapolated a finite
truncated density outside the restricted latent support. The test was then
narrowed to bounded treatment after verifying that `float_max` is the exact
invertible positive endpoint on this platform, not an outside-support value.

### Implementation decisions

- Removed all clipping from positive and bounded forward assignment transforms.
- Froze an absolute latent interval for each transformed domain. Positive bounds
  are the logs of the smallest/largest representable positive Float64 values.
  Bounded candidates come from the exact inverse of the nearest interior
  treatment values; a fixed bisection against latent zero locates the closest
  side that forward-transforms strictly inside open support.
- Conditioned each confounded/randomized Gaussian component independently on
  that same latent interval. `_log_prob` subtracts the corresponding
  location-dependent log normalizer before stable mixture combination and the
  exact Jacobian.
- Implemented stable standard-normal log-CDF/log-interval calculations without
  SciPy, including asymptotic lower-tail evaluation, and inverse-CDF sampling by
  100-step bisection in absolute latent coordinates. The sampler returns an
  interior bisection midpoint and performs no clipping.
- Restricted `get_grid` to the intersection of the original central Gaussian
  region and the invertible latent support.
- Treatments inside the mathematical bounded domain but outside the restricted
  assignment image now receive `-inf` from `log_prob`. Outcome interventions
  remain valid across the full treatment domain.
- Updated the round-1 support-edge test to target the final restricted-law edges;
  its old nearest-domain-edge expectation was incompatible with the redesigned
  exact truncated law.

The first bounded-support implementation tried walking inward one latent ULP at
a time. A targeted test run was interrupted after 34 seconds because a broad
rounding plateau made that algorithm effectively nonterminating. It was replaced
with the fixed bisection described above; no such loop remains.

### GREEN evidence

The round-trip and exact truncated-mixture tests after implementation:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py::test_transforms_round_trip_on_restricted_invertible_latent_support tests/datasets/test_semi_synthetic_continuous_dgp.py::test_positive_log_density_matches_exact_truncated_latent_mixture -q
```

Result: `3 passed in 1.27s`.

The bounded outside-support likelihood regression after correction:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py::test_bounded_log_density_is_negative_infinity_outside_restricted_latent_support -q
```

Result: `1 passed in 0.99s`.

Final continuous file after removing the last latent-space defensive clip:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py -q
```

Result: `16 passed in 2.62s`.

### Final verification

```text
uv run ruff format src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
uv run ruff check src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py -q
```

Result: `2 files left unchanged`; `All checks passed!`; `32 passed in 1.68s`.

```text
uv run pytest tests/datasets/test_all_datasets.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_regressor.py tests/datasets/test_semi_synthetic_classifier.py -q
uv run pytest -q
```

Result: `137 passed in 1.91s`; full suite `774 passed, 419 warnings in 7.55s`.
Warning classes match the existing baseline.

## Fix round 3 — narrow standardized truncation intervals

### Root cause and RED evidence

The crossing-zero branch of `_standard_normal_log_interval` computed
`1 - (left_tail + right_tail)` in ordinary probability space. When latent
support width divided by treatment noise was below machine epsilon, both tails
rounded to `0.5`; the interval log mass became `-inf`. Sampling then collapsed
to a support edge and component likelihoods used a corrupted normalizer.

A parameterized bounded/positive regression was added first. The final RED
fixture uses bounded noise `1e18` and positive noise `1e19`, ensuring both
standardized support widths are below Float64 epsilon:

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py::test_narrow_standardized_support_has_finite_uniform_limit_and_nontrivial_samples -q
```

Result before the fix: `2 failed, 2 warnings in 1.05s`. The bounded sample had
one unique inverse treatment across all eight rows. The positive sample varied,
but its log-density missed the hand-derived uniform limit by `0.64902786`.
Both paths emitted divide-by-zero from the rounded crossing-zero subtraction.

### Minimal implementation

For finite intervals satisfying

```text
width * max(1, abs(midpoint)) <= sqrt(machine epsilon)
```

the shared interval primitive now evaluates the midpoint log integral directly:

```text
log(width) - midpoint**2 / 2 - log(sqrt(2*pi))
```

In this activation regime, variation of the standard-normal log density across
the interval is at most `sqrt(epsilon)`, so midpoint quadrature error is at
machine precision. This avoids subtracting rounded CDF/tail values and is used
unchanged by both inverse-CDF sampling and truncated-component likelihoods.
Ordinary-width and one-sided-tail branches are untouched; no dependency was
added.

### GREEN and final verification

```text
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py::test_narrow_standardized_support_has_finite_uniform_limit_and_nontrivial_samples -q
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py -q
```

Results: `2 passed in 1.22s`; `18 passed in 1.60s`. Both domains produce more
than four distinct inverse treatments with nontrivial quantile spread, finite
densities, and the uniform-limit log-density within absolute tolerance `1e-12`.

```text
uv run ruff format src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
uv run ruff check src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py -q
```

Result: `2 files left unchanged`; `All checks passed!`; `34 passed in 1.61s`.

```text
uv run pytest tests/datasets/test_all_datasets.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_regressor.py tests/datasets/test_semi_synthetic_classifier.py -q
uv run pytest -q
```

Result: `137 passed in 1.86s`; full suite `776 passed, 419 warnings in 5.51s`.
Warning classes match the existing baseline.
