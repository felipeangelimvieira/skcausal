# Task 3 report — shared probability and calibration numerics

## Files changed

- `src/skcausal/datasets/semi_synthetic.py`
  - Added private stable helpers: `_softmax`, `_normal_logpdf`, `_log_mixture`,
    and `_normalized_log_weights`.
  - Added `_calibrate_softmax_intercepts`, a damped Newton solver with the final
    category held at the reference intercept.
  - Added `_calibrate_confounding_strength`, which expands the upper bracket up
    to 64 times and then bisects the target crossing.
- `tests/datasets/test_semi_synthetic_helpers.py`
  - Added behavioral tests for empirical softmax calibration, stable mixture
    probabilities/normalization, attainable and unattainable strength targets,
    and a target reached by the 64th allowed bracket expansion.

No public API or diagnostics were added.

## Decisions

- Softmax calibration solves only the first `K - 1` mean-probability residuals;
  the last intercept remains zero as the identifiability reference.
- Newton updates use backtracking damping and reject singular or nonfinite
  calculations with `ValueError`.
- Strength calibration accepts either monotonic crossing direction, reports the
  evaluated metric interval in non-attainment/convergence errors, and returns
  exact endpoint matches without unnecessary bisection.

## TDD evidence

RED:

```text
$ uv run pytest tests/datasets/test_semi_synthetic_helpers.py -q
ImportError: cannot import name '_calibrate_confounding_strength'
```

The prescribed new imports were absent. A subsequent boundary regression test
also failed before its fix with:

```text
ValueError: Target is outside the attainable metric interval [0.0, 64.0].
```

GREEN:

```text
$ uv run pytest tests/datasets/test_semi_synthetic_helpers.py -q
9 passed in 0.83s

$ uv run pytest tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_base.py -q
15 passed in 1.22s
```

## Verification

```text
$ uv run ruff check src/skcausal/datasets/semi_synthetic.py tests/datasets/test_semi_synthetic_helpers.py
All checks passed!

$ uv run ruff format --check src/skcausal/datasets/semi_synthetic.py tests/datasets/test_semi_synthetic_helpers.py
2 files already formatted

$ uv run pytest -q
732 passed, 419 warnings in 4.16s
```

## Concerns

`_calibrate_confounding_strength` follows the plan's crossing-and-bisection
contract, so callers must provide a scalar metric that crosses the target over
the expanded interval. Later categorical and continuous DGPs are responsible
for choosing their oracle metrics and validating their public parameters.

## Fix round 1 — large initial bisection bracket

The fixed 100-iteration bisection cap could reject a valid target when the
already-bracketed initial upper strength was very large. The bisection budget
now uses `ceil(log2(bracket_width / 1e-6)) + 1`, with 100 retained as a
minimum, so it permits the stated absolute-width convergence condition to be
checked after sufficient halvings.

RED:

```text
$ uv run pytest tests/datasets/test_semi_synthetic_helpers.py -q
1 failed, 9 passed in 2.42s

ValueError: Target did not converge within the attainable metric interval
[0.0, 7.888609052210118e+29].
```

The new regression called the real calibrator with `metric=lambda value: value`,
`target=0.5`, and `initial_strength=1e60`.

GREEN:

```text
$ uv run pytest tests/datasets/test_semi_synthetic_helpers.py -q
10 passed in 1.99s

$ uv run pytest tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_base.py -q
16 passed in 0.95s

$ uv run ruff check src/skcausal/datasets/semi_synthetic.py tests/datasets/test_semi_synthetic_helpers.py
All checks passed!

$ uv run ruff format --check src/skcausal/datasets/semi_synthetic.py tests/datasets/test_semi_synthetic_helpers.py
2 files already formatted

$ uv run pytest -q
733 passed, 419 warnings in 4.35s
```
