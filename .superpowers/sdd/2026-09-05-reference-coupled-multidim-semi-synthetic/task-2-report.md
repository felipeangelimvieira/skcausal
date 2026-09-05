# Task 2 Report: Fixed Multisource Dataset and Mean-Regressor Confounding

## Implementation summary

- Replaced the sampled multidimensional treatment DGP with fixed treatments assembled directly from selected source outcomes.
- Couples complete source rows through the existing Gaussian-copula orders, records final output-to-source row indices, and exposes requested Pearson/Spearman diagnostics.
- Fits cloned ordinary sklearn mean regressors on each source's original-name pandas frame, combines standardized predictions with equal or Dirichlet weights, and uses their square-root-weighted sum as the baseline outcome.
- Adds centered additive B-spline treatment effects, fixed sampling hooks, treatment grids, and the required global dataset parameters.
- Replaced obsolete density/assignment tests with public fixed-treatment, diagnostics, seeded-Dirichlet, regressor-cloning, and counterfactual spline tests. The existing toy fixtures already contain the required categorical covariates, so no test-utility edit was needed.

## Files changed

- `src/skcausal/datasets/semi_synthetic_multidim.py`
- `tests/datasets/test_semi_synthetic_multidim_dgp.py`
- `.superpowers/sdd/2026-09-05-reference-coupled-multidim-semi-synthetic/task-2-report.md`

## Self-review

- Reviewed the complete diff from `f4e44e8`; it removes the obsolete component broadcasting, density, assignment, calibration, and categorical-treatment implementation instead of retaining compatibility paths.
- Confirmed fixed treatments, covariate rows, and `source_row_indices_` share the same copula-derived order.
- Confirmed regressors are cloned and receive source-local pandas frames with original column names, preserving standard sklearn preprocessing pipelines.
- `git diff --check f4e44e8` passed.
- `uv run ruff check src/skcausal/datasets/semi_synthetic_multidim.py tests/datasets/test_semi_synthetic_multidim_dgp.py` passed.

## Concerns

- The focused test run emits third-party pandas and existing base-layer deprecation warnings. The requested Polars horizontal-concatenation warning is absent. The full suite has its existing warning inventory but no failures.

## TDD evidence

### RED

Command:

```text
uv run pytest tests/datasets/test_semi_synthetic_multidim_dgp.py -q -k "source_outcomes or diagnostics or dirichlet or spline_response"
```

Output:

```text
FFFF
TypeError: MultidimSemiSyntheticDataset.__init__() got an unexpected keyword argument 'regressors'
4 failed, 5 deselected in 1.30s
```

### GREEN

Command:

```text
uv run pytest tests/datasets/test_semi_synthetic_multidim_dgp.py tests/datasets/test_all_datasets.py -q
```

Output:

```text
141 passed, 17 warnings in 2.79s
```

### Full-suite verification

Command:

```text
uv run pytest -q
```

Output:

```text
856 passed, 655 warnings in 5.74s
```
