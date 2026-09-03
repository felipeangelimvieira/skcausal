# Multi-Dimensional Semi-Synthetic DGP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `MultidimSemiSyntheticDataset`, a vector-treatment semi-synthetic DGP driven by a frozen prognostic score on the real outcome, with per-component confounding loadings, exchangeable noise correlation, optional feature mimicking, optional ordered-probit discretization, and exact oracles.

**Architecture:** One new module `semi_synthetic_multidim.py` subclasses `BaseSemiSyntheticDataset`. The base gains three small extensions: a `_split_source` hook so a subclass can keep source columns and release a reduced `X`, a ridge-fitted `confounders` role in the causal score map, and user-supplied confounder coefficients in the baseline surface. One-dimensional marginal-quantile helpers move from the scalar continuous DGP to module level so both classes share them. The bias metric is Monte Carlo under the marginal with frozen common random numbers and an exactly enumerated discrete part, so the existing ladder-plus-Brent strength search is reused unchanged.

**Tech Stack:** Python, NumPy, SciPy (`log_ndtr`, `ndtr`, `ndtri`, `logsumexp`, `stats.multivariate_normal` in tests), Polars, scikit-learn (existing encoders), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-02-multidim-semi-synthetic-dgp-design.md`

## Global Constraints

- Work in the worktree `~/orca/workspaces/skcausal/manatee` on branch `felipeangelimvieira/synth-observational-data-generator`. It already holds uncommitted revisions to the semi-synthetic modules; do not revert, stash, or reformat them. Commit only the files each task names.
- Run tests as `PYTHONPATH=src python -m pytest ...` from the worktree root. A bare `import skcausal` resolves to a different checkout.
- `tests/datasets/test_all_datasets.py`, the density-estimator tests, and the "all objects" estimator tests fail to collect in this environment (`polars has no attribute Float16`, skbase import errors). This is pre-existing; run suites with `--ignore=tests/datasets/test_all_datasets.py` and do not try to fix it.
- The categorical and scalar-continuous reference DGPs must keep their behavior; their existing tests are the regression guard for every base change.
- `ruff check` and `ruff format --check` must pass on every file you touch (`ruff` is installed in the environment).
- Continuous components live on the real line only. `treatment_correlation` must be in `[0, 1)` and must be zero when two or more components are categorical.
- Released treatment columns are named `t_0`, `t_1`, ... in component order. Categorical levels are the strings `"0"` to `"K-1"`.
- At `confounding_strength=0` and `treatment_only_strength=0` the oracle confounding bias must be exactly zero.
- The outcome depends only on released treatment values, never on the latent behind a categorical component.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01GLGT1kUJvvubxK8XBvh5uB
  ```

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/skcausal/datasets/semi_synthetic.py` (modify) | Base class and shared helpers. Gains `_split_source`, `_ridge_projections`, `fitted_confounders` in `_fit_causal_score_map`, `confounder_coefficients` in `_fit_baseline_surface`, pass-through kwargs on `_fit_causal_structure`, and the lifted marginal helpers `_log_mixture_probability`, `_gaussian_mixture_marginal_cdf`, `_bisect_marginal_quantiles`. |
| `src/skcausal/datasets/semi_synthetic_continuous.py` (modify) | Uses the lifted marginal helpers; behavior unchanged. |
| `src/skcausal/datasets/semi_synthetic_multidim.py` (create) | Pure helpers (per-component validation, exchangeable correlation, interval log-mass, Gaussian log-density, Gaussian conditioning, numeric column extraction) and `MultidimSemiSyntheticDataset`. |
| `src/skcausal/datasets/__init__.py` (modify) | Export the new class. |
| `tests/datasets/test_semi_synthetic_helpers.py` (modify) | Tests for the base helper extensions. |
| `tests/datasets/test_semi_synthetic_base.py` (modify) | Test for the `_split_source` hook. |
| `tests/datasets/test_semi_synthetic_multidim_dgp.py` (create) | Tests for the new module. |
| `tests/test_lookup.py` (modify) | Discovery of the new class. |
| `docs/research/semi_synthetic_datasets.qmd` (modify) | New step for multi-dimensional and mixed treatments; configuration reference rows. |

---

### Task 1: Base hooks — source split, ridge-fitted confounders, fixed surface coefficients

**Files:**
- Modify: `src/skcausal/datasets/semi_synthetic.py` (dataclass `_CausalScoreMap` ~line 38, `_fit_causal_score_map` ~line 173, `_fit_baseline_surface` ~line 286, `BaseSemiSyntheticDataset._prepare` ~line 527, `_fit_causal_structure` ~line 623)
- Test: `tests/datasets/test_semi_synthetic_helpers.py`, `tests/datasets/test_semi_synthetic_base.py`

**Interfaces:**
- Produces: `_ridge_projections(library, targets) -> (projection (p, d) ndarray, r2 (d,) ndarray)`.
- Produces: `_fit_causal_score_map(X, rng, fitted_confounders=None)`; when given an `(n, d)` float array (NaN allowed), the `confounders` role has `d` ridge-fitted columns and `score_map.fitted_confounder_r2` is the `(d,)` R² array (else `None`).
- Produces: `_fit_baseline_surface(scores, rng, confounder_coefficients=None)`; when given, those coefficients are used verbatim.
- Produces: `BaseSemiSyntheticDataset._split_source(self, X, treatment, outcome) -> X_released` (default identity), called by `_prepare` after validating the loaded `X` and before `n`, `_covariates`, and `_prepare_dgp`.
- Produces: `BaseSemiSyntheticDataset._fit_causal_structure(self, X, rng, *, fitted_confounders=None, confounder_coefficients=None)`.

- [ ] **Step 1: Write the failing helper tests**

Append to `tests/datasets/test_semi_synthetic_helpers.py` (add `_ridge_projections` to the existing import list from `skcausal.datasets.semi_synthetic`):

```python
def test_ridge_projections_recover_linear_targets_and_skip_missing_rows():
    rng = np.random.default_rng(3)
    library = rng.normal(size=(200, 6))
    library[:, 5] = 2.0  # zero-variance column must get a zero coefficient
    targets = np.column_stack(
        (
            3.0 * library[:, 0] - 1.5 * library[:, 2] + 0.5,
            library[:, 1] + rng.normal(scale=0.1, size=200),
        )
    )
    targets[:10, 1] = np.nan

    projection, r2 = _ridge_projections(library, targets)

    assert projection.shape == (6, 2)
    assert projection[5].tolist() == [0.0, 0.0]
    assert r2.shape == (2,)
    assert r2[0] > 0.999
    assert 0.9 < r2[1] <= 1.0
    fitted = library @ projection[:, 0]
    fitted = fitted - fitted.mean()
    np.testing.assert_allclose(fitted, targets[:, 0] - targets[:, 0].mean(), atol=0.15)


def test_ridge_projections_require_two_finite_targets():
    library = np.random.default_rng(0).normal(size=(5, 2))
    with pytest.raises(ValueError, match="at least two finite"):
        _ridge_projections(library, np.array([1.0, np.nan, np.nan, np.nan, np.nan]))


def test_score_map_accepts_fitted_confounders():
    X = ToyRealDataset(include_missing=True).load()[0]
    targets = np.column_stack(
        (
            np.linspace(-1.0, 1.0, X.height),
            X.get_column("income").to_numpy().astype(float),
        )
    )
    score_map, scores = _fit_causal_score_map(
        X, np.random.default_rng(4), fitted_confounders=targets
    )

    assert scores.confounders.shape == (X.height, 2)
    np.testing.assert_allclose(scores.confounders.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(scores.confounders.std(axis=0), 1.0, atol=1e-10)
    assert score_map.fitted_confounder_r2.shape == (2,)
    assert np.all((score_map.fitted_confounder_r2 >= 0.0) & (score_map.fitted_confounder_r2 <= 1.0))
    np.testing.assert_allclose(score_map.transform(X).confounders, scores.confounders)
    assert scores.treatment_only.shape == (X.height, 2)


def test_baseline_surface_accepts_fixed_confounder_coefficients():
    X = ToyRealDataset().load()[0]
    _, scores = _fit_causal_score_map(X, np.random.default_rng(8))
    surface = _fit_baseline_surface(
        scores, np.random.default_rng(9), confounder_coefficients=[1.0, 0.5, -0.25]
    )
    np.testing.assert_allclose(surface.confounder_coefficients, [1.0, 0.5, -0.25])
    mean, _ = surface.evaluate(scores)
    np.testing.assert_allclose(mean.std(ddof=0), 1.0, atol=1e-12)

    with pytest.raises(ValueError, match="confounder_coefficients"):
        _fit_baseline_surface(
            scores, np.random.default_rng(9), confounder_coefficients=[1.0, 0.5]
        )
```

Append to `tests/datasets/test_semi_synthetic_base.py`:

```python
class SplittingSemiSyntheticDataset(ToySemiSyntheticDataset):
    def _split_source(self, X, treatment, outcome):
        self.seen_outcome_columns_ = list(outcome.columns)
        return X.drop("income")

    def _sample_treatment(self, X, rng):
        return pl.DataFrame({"t": X.get_column("age").to_numpy() / 50.0})

    def _log_prob(self, X, treatment):
        return np.zeros(X.height)


def test_base_split_source_hook_releases_reduced_covariates():
    dataset = SplittingSemiSyntheticDataset(ToyRealDataset(), random_state=1)
    X, treatment, _ = dataset.load()

    assert list(X.columns) == ["age", "region"]
    assert dataset.seen_outcome_columns_ == ["source_y"]
    assert dataset.n == 8
    assert treatment.shape == (8, 1)
    assert dataset.log_prob(X.to_numpy(), treatment).shape == (8, 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_base.py -v`
Expected: FAIL. The helper file fails at import (`_ridge_projections` missing); the base test fails because `_split_source` is never called (columns still include `income`).

- [ ] **Step 3: Add `_ridge_projections` and the `fitted_confounders` path**

In `src/skcausal/datasets/semi_synthetic.py`, add a `fitted_confounder_r2` field to the dataclass:

```python
@dataclass
class _CausalScoreMap:
    ...
    residualization_order: int
    residualization: np.ndarray
    fitted_confounder_r2: np.ndarray | None = None
```

Add this function directly above `_fit_causal_score_map`:

```python
_RIDGE_PENALTY_FRACTION = 0.01


def _ridge_projections(library, targets):
    """Ridge-fit each target column on the nonlinear library.

    Returns a ``(p, d)`` projection that maps the raw library to a prediction
    equal to the centered ridge fit up to an additive constant, and the ``(d,)``
    in-sample R² of each fit. Rows whose target is not finite are excluded from
    the fit. The penalty is ``0.01 * n_observed`` on library columns
    standardized to unit variance, so shrinkage is scale-free; zero-variance
    columns receive a zero coefficient.
    """

    library = np.asarray(library, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets[:, None]
    projection = np.zeros((library.shape[1], targets.shape[1]))
    r2 = np.zeros(targets.shape[1])
    for index in range(targets.shape[1]):
        observed = np.isfinite(targets[:, index])
        if observed.sum() < 2:
            raise ValueError(
                "Fitted confounder targets need at least two finite values."
            )
        design = library[observed]
        center = design.mean(axis=0)
        scale = design.std(axis=0)
        active = scale > 0.0
        standardized = (design[:, active] - center[active]) / scale[active]
        target = targets[observed, index]
        target = target - target.mean()
        penalty = _RIDGE_PENALTY_FRACTION * observed.sum()
        gram = standardized.T @ standardized + penalty * np.eye(int(active.sum()))
        weights = np.linalg.solve(gram, standardized.T @ target)
        projection[active, index] = weights / scale[active]
        residual = target - standardized @ weights
        total = float(target @ target)
        r2[index] = 1.0 - float(residual @ residual) / total if total > 0.0 else 0.0
    return projection, r2
```

Change the signature of `_fit_causal_score_map` to `def _fit_causal_score_map(X, rng, fitted_confounders=None):` and replace the role loop:

```python
    projections = {}
    raw = {}
    fitted_r2 = None
    for name, dimension in _ROLE_DIMENSIONS.items():
        if name == "confounders" and fitted_confounders is not None:
            projection, fitted_r2 = _ridge_projections(library, fitted_confounders)
            projections[name] = projection
            raw[name] = library @ projection
            continue
        projection = np.zeros((library.shape[1], dimension))
        for index in range(dimension):
            n_terms = min(5, library.shape[1])
            nonlinear = rng.integers(n_encoded, library.shape[1])
            remaining = np.delete(np.arange(library.shape[1]), nonlinear)
            selected = np.concatenate(
                (
                    np.array([nonlinear]),
                    rng.choice(remaining, size=n_terms - 1, replace=False),
                )
            )
            projection[selected, index] = rng.normal(size=selected.size)
        projections[name] = projection
        raw[name] = library @ projection
```

and pass `fitted_confounder_r2=fitted_r2` when constructing `_CausalScoreMap`.

- [ ] **Step 4: Add `confounder_coefficients` to `_fit_baseline_surface`**

Replace the head of `_fit_baseline_surface`:

```python
def _fit_baseline_surface(scores, rng, confounder_coefficients=None):
    n_confounders = scores.confounders.shape[1]
    prognostic_scale = 1.0 / np.sqrt(scores.prognostic.shape[1])
    if confounder_coefficients is None:
        confounder_coefficients = rng.normal(
            scale=1.0 / np.sqrt(n_confounders), size=n_confounders
        )
    else:
        confounder_coefficients = np.asarray(confounder_coefficients, dtype=float)
        if confounder_coefficients.shape != (n_confounders,) or not np.isfinite(
            confounder_coefficients
        ).all():
            raise ValueError(
                "confounder_coefficients must be a finite vector with one entry "
                "per confounder score."
            )
    prognostic_coefficients = rng.normal(
        scale=prognostic_scale, size=scores.prognostic.shape[1]
    )
```

The rest of the function is unchanged.

- [ ] **Step 5: Add the `_split_source` hook and pass-through kwargs**

In `BaseSemiSyntheticDataset._prepare`, replace the loading block:

```python
        self.real_dataset_ = self.real_dataset.clone()
        X, treatment, outcome = self.real_dataset_.load()
        X = self._coerce_backend_frame(X, backend="polars")
        if X.height == 0 or len(set(X.columns)) != X.width:
            raise ValueError(
                "real_dataset must provide nonempty covariates with unique columns."
            )
        X = self._coerce_backend_frame(
            self._split_source(X, treatment, outcome), backend="polars"
        )
        if X.width == 0:
            raise ValueError(
                "At least one covariate column must remain after the source split."
            )
        self.n = X.height
```

Add the hook next to `_check_and_transform_X`:

```python
    def _split_source(self, X, treatment, outcome):
        """Return the covariates to release; subclasses may keep source columns.

        Called once during preparation with the loaded Polars covariates and
        the raw source treatment and outcome frames. The default releases
        ``X`` unchanged.
        """

        return X
```

Replace `_fit_causal_structure`'s head:

```python
    def _fit_causal_structure(
        self, X, rng, *, fitted_confounders=None, confounder_coefficients=None
    ):
        """Freeze the score map and standardized baseline outcome surface."""

        self.score_map_, self.source_scores_ = _fit_causal_score_map(
            X, rng, fitted_confounders=fitted_confounders
        )
        self.baseline_surface_ = _fit_baseline_surface(
            self.source_scores_, rng, confounder_coefficients=confounder_coefficients
        )
```

The rest of the method is unchanged.

- [ ] **Step 6: Run the tests to verify they pass, plus the reference DGP suites**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_continuous_dgp.py -q`
Expected: all PASS. Then `ruff check src/skcausal/datasets/semi_synthetic.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_base.py && ruff format --check` on the same files: clean.

- [ ] **Step 7: Commit**

```bash
git add src/skcausal/datasets/semi_synthetic.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_base.py
git commit -m "feat(semi-synthetic): source split hook, ridge-fitted confounders, fixed surface coefficients"
```

---

### Task 2: Lift the one-dimensional marginal helpers to module level

**Files:**
- Modify: `src/skcausal/datasets/semi_synthetic.py` (add three functions after `_normalized_log_weights`)
- Modify: `src/skcausal/datasets/semi_synthetic_continuous.py` (`_latent_marginal_cdf` ~line 479, `_latent_marginal_quantiles` ~line 513, remove `_log_mixture_probability` ~line 676)
- Test: `tests/datasets/test_semi_synthetic_helpers.py`

**Interfaces:**
- Produces: `_log_mixture_probability(confounded, randomized, weight)` (moved from the continuous module, same body).
- Produces: `_gaussian_mixture_marginal_cdf(latent (m,), means (n,), scale, weight) -> (m,)`: the CDF of the row-averaged mixture `(1 - w) * mean_i N(means_i, scale^2) + w * N(0, scale^2)`.
- Produces: `_bisect_marginal_quantiles(cdf, probabilities, lower, upper) -> ndarray`: vectorized bisection of a monotone `cdf` callable to the requested probabilities on `[lower, upper]`.

- [ ] **Step 1: Write the failing helper tests**

Append to `tests/datasets/test_semi_synthetic_helpers.py` (add `_bisect_marginal_quantiles`, `_gaussian_mixture_marginal_cdf` to the import list; add `from scipy.special import ndtr` at the top):

```python
def test_gaussian_mixture_marginal_cdf_and_quantiles_agree_with_closed_form():
    means = np.array([-1.0, 0.5, 2.0])
    scale, weight = 1.5, 0.25
    latent = np.array([-3.0, 0.0, 1.0, 4.0])

    expected = (1.0 - weight) * ndtr(
        (latent[:, None] - means[None, :]) / scale
    ).mean(axis=1) + weight * ndtr(latent / scale)
    np.testing.assert_allclose(
        _gaussian_mixture_marginal_cdf(latent, means, scale, weight), expected
    )

    probabilities = np.array([0.1, 0.5, 0.9])
    quantiles = _bisect_marginal_quantiles(
        lambda value: _gaussian_mixture_marginal_cdf(value, means, scale, weight),
        probabilities,
        -20.0,
        20.0,
    )
    np.testing.assert_allclose(
        _gaussian_mixture_marginal_cdf(quantiles, means, scale, weight),
        probabilities,
        atol=1e-9,
    )
    assert np.all(np.diff(quantiles) > 0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/datasets/test_semi_synthetic_helpers.py -q`
Expected: FAIL at import (`_gaussian_mixture_marginal_cdf` missing).

- [ ] **Step 3: Add the helpers to the base module**

In `src/skcausal/datasets/semi_synthetic.py`, extend the SciPy import to `from scipy.special import logsumexp, ndtr` and add after `_normalized_log_weights`:

```python
def _log_mixture_probability(confounded, randomized, weight):
    if weight == 0.0:
        return confounded
    if weight == 1.0:
        return randomized
    return (1.0 - weight) * confounded + weight * randomized


def _gaussian_mixture_marginal_cdf(latent, means, scale, weight):
    """CDF of the row-averaged latent mixture at each ``latent`` value.

    The marginal is ``(1 - weight) * mean_i N(means_i, scale^2)`` plus
    ``weight * N(0, scale^2)``.
    """

    latent = np.asarray(latent, dtype=float)
    means = np.asarray(means, dtype=float)
    confounded = ndtr((latent[:, None] - means[None, :]) / scale).mean(axis=1)
    randomized = ndtr(latent / scale)
    return _log_mixture_probability(confounded, randomized, weight)


def _bisect_marginal_quantiles(cdf, probabilities, lower, upper):
    """Vectorized bisection of a monotone ``cdf`` on ``[lower, upper]``."""

    probabilities = np.asarray(probabilities, dtype=float)
    low = np.full(probabilities.shape, float(lower))
    high = np.full(probabilities.shape, float(upper))
    for _ in range(200):
        middle = 0.5 * low + 0.5 * high
        below = cdf(middle) < probabilities
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
        if np.all(high - low <= 1e-12 * np.maximum(1.0, np.abs(middle))):
            break
    return 0.5 * low + 0.5 * high
```

- [ ] **Step 4: Point the continuous DGP at the lifted helpers**

In `src/skcausal/datasets/semi_synthetic_continuous.py`:

- Extend the base import to include `_bisect_marginal_quantiles`, `_gaussian_mixture_marginal_cdf`, `_log_mixture_probability`.
- Delete the module-level `_log_mixture_probability` at the bottom of the file.
- In `_latent_marginal_cdf`, replace the `if self.treatment_domain == "real":` branch so the whole method reads:

```python
    def _latent_marginal_cdf(self, latent, means):
        scale = self.treatment_noise_scale
        if self.treatment_domain == "real":
            return _gaussian_mixture_marginal_cdf(
                latent, means, scale, self.randomized_weight
            )
        latent = np.asarray(latent, dtype=float)[:, None]
        lower, upper = self.latent_support_bounds_
        confounded_mass, randomized_mass = self._component_log_masses(means)
        confounded = _interval_cdf(
            (latent - means[None, :]) / scale,
            ((lower - means) / scale)[None, :],
            ((upper - means) / scale)[None, :],
            confounded_mass[None, :],
        )
        randomized = _interval_cdf(
            latent / scale, lower / scale, upper / scale, randomized_mass
        )
        return _log_mixture_probability(
            confounded.mean(axis=1), randomized[:, 0], self.randomized_weight
        )
```

- Replace `_latent_marginal_quantiles` with:

```python
    def _latent_marginal_quantiles(self, probabilities, means):
        lower, upper = self._latent_marginal_range(means)
        return _bisect_marginal_quantiles(
            lambda value: self._latent_marginal_cdf(value, means),
            probabilities,
            lower,
            upper,
        )
```

- [ ] **Step 5: Run the helper and continuous suites**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_continuous_dgp.py -q`
Expected: all PASS. Then `ruff check` and `ruff format --check` on the two source files and the test file: clean.

- [ ] **Step 6: Commit**

```bash
git add src/skcausal/datasets/semi_synthetic.py src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_helpers.py
git commit -m "refactor(semi-synthetic): lift latent marginal cdf and quantile helpers"
```

---

### Task 3: Multidim pure helpers

**Files:**
- Create: `src/skcausal/datasets/semi_synthetic_multidim.py`
- Create: `tests/datasets/test_semi_synthetic_multidim_dgp.py`

**Interfaces:**
- Produces: `_per_component(name, value, n_components, *, default=None) -> list` (scalar broadcast or validated length-k sequence; strings count as scalars).
- Produces: `_validate_loadings(value, k) -> ndarray (k,)`, `_validate_levels(value, k) -> list[int | None]`, `_validate_targets(value, k) -> list[str | None]`.
- Produces: `_exchangeable_correlation(k, rho) -> ndarray (k, k)`.
- Produces: `_interval_log_mass(lower, upper) -> ndarray`: `log(Phi(upper) - Phi(lower))` for standardized bounds, `±inf` allowed.
- Produces: `_gaussian_logpdf(residual (..., d), cholesky (d, d)) -> (...)`: log-density of `residual` under `N(0, L L')`; returns zeros when `d == 0`.
- Produces: `_conditional_gaussian(covariance, continuous_idx, categorical_idx) -> (regression (kd, kc), conditional_covariance (kd, kd))`.
- Produces: `_numeric_column(frame, name, label) -> ndarray (n,)` float with NaN for nulls; raises for non-numeric or fewer than two distinct finite values.

- [ ] **Step 1: Write the failing tests**

Create `tests/datasets/test_semi_synthetic_multidim_dgp.py`:

```python
import numpy as np
import polars as pl
import pytest
from scipy.special import ndtr
from scipy.stats import multivariate_normal

from skcausal.datasets.semi_synthetic_multidim import (
    _conditional_gaussian,
    _exchangeable_correlation,
    _gaussian_logpdf,
    _interval_log_mass,
    _numeric_column,
    _per_component,
    _validate_levels,
    _validate_loadings,
    _validate_targets,
)
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


def test_per_component_broadcasts_scalars_and_checks_lengths():
    assert _per_component("x", 2.0, 3) == [2.0, 2.0, 2.0]
    assert _per_component("x", None, 2, default=1.0) == [1.0, 1.0]
    assert _per_component("x", "income", 2) == ["income", "income"]
    assert _per_component("x", [None, "a"], 2) == [None, "a"]
    with pytest.raises(ValueError, match="length n_treatments=2"):
        _per_component("x", [1.0, 2.0, 3.0], 2)


def test_component_validators():
    np.testing.assert_allclose(_validate_loadings(None, 2), [1.0, 1.0])
    np.testing.assert_allclose(_validate_loadings([1.0, 0.0], 2), [1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        _validate_loadings([1.0, np.inf], 2)

    assert _validate_levels(None, 2) == [None, None]
    assert _validate_levels([None, 3], 2) == [None, 3]
    with pytest.raises(ValueError, match="at least 2"):
        _validate_levels([1, None], 2)
    with pytest.raises(TypeError, match="integers"):
        _validate_levels([True, None], 2)

    assert _validate_targets(None, 2) == [None, None]
    assert _validate_targets([None, "income"], 2) == [None, "income"]
    with pytest.raises(ValueError, match="repeat"):
        _validate_targets(["age", "age"], 2)
    with pytest.raises(TypeError, match="column names"):
        _validate_targets([1, None], 2)


def test_exchangeable_correlation_is_symmetric_with_unit_diagonal():
    matrix = _exchangeable_correlation(3, 0.4)
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    np.testing.assert_allclose(matrix[np.triu_indices(3, 1)], 0.4)
    np.linalg.cholesky(matrix)


def test_interval_log_mass_matches_direct_formula_and_tails():
    lower = np.array([-np.inf, -1.0, 0.5, -np.inf, 30.0, -0.2])
    upper = np.array([-0.5, 0.25, np.inf, np.inf, np.inf, 0.2])
    expected = np.log(ndtr(upper) - ndtr(lower))
    expected[4] = np.log(ndtr(-30.0))
    np.testing.assert_allclose(_interval_log_mass(lower, upper), expected, rtol=1e-12)
    # Deep tail below the double range of ndtr differences.
    assert np.isfinite(_interval_log_mass(np.array([-np.inf]), np.array([-40.0])))[0]


def test_gaussian_logpdf_matches_scipy_and_handles_empty_dimension():
    covariance = np.array([[2.0, 0.6], [0.6, 1.0]])
    cholesky = np.linalg.cholesky(covariance)
    residual = np.random.default_rng(1).normal(size=(4, 3, 2))
    expected = multivariate_normal(mean=np.zeros(2), cov=covariance).logpdf(
        residual.reshape(-1, 2)
    ).reshape(4, 3)
    np.testing.assert_allclose(_gaussian_logpdf(residual, cholesky), expected)
    assert _gaussian_logpdf(np.zeros((5, 0)), np.zeros((0, 0))).shape == (5,)
    assert np.all(_gaussian_logpdf(np.zeros((5, 0)), np.zeros((0, 0))) == 0.0)


def test_conditional_gaussian_matches_textbook_formula():
    covariance = _exchangeable_correlation(3, 0.3) * 4.0
    continuous = np.array([0, 2])
    categorical = np.array([1])
    regression, conditional = _conditional_gaussian(covariance, continuous, categorical)

    cov_cc = covariance[np.ix_(continuous, continuous)]
    cov_dc = covariance[np.ix_(categorical, continuous)]
    expected_regression = cov_dc @ np.linalg.inv(cov_cc)
    np.testing.assert_allclose(regression, expected_regression)
    np.testing.assert_allclose(
        conditional,
        covariance[np.ix_(categorical, categorical)]
        - expected_regression @ cov_dc.T,
    )

    regression, conditional = _conditional_gaussian(covariance, np.array([0, 1, 2]), np.array([], dtype=int))
    assert regression.shape == (0, 3) and conditional.shape == (0, 0)
    regression, conditional = _conditional_gaussian(covariance, np.array([], dtype=int), np.array([1]))
    assert regression.shape == (1, 0)
    np.testing.assert_allclose(conditional, [[4.0]])


def test_numeric_column_extracts_floats_and_rejects_bad_columns():
    X = ToyRealDataset(include_missing=True).load()[0]
    values = _numeric_column(X, "age", "target column 'age'")
    assert values.shape == (8,)
    assert np.isnan(values[2])
    with pytest.raises(ValueError, match="must be numeric"):
        _numeric_column(X, "region", "target column 'region'")
    with pytest.raises(ValueError, match="two distinct finite"):
        _numeric_column(
            X.with_columns(pl.lit(1.0).alias("age")), "age", "target column 'age'"
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/datasets/test_semi_synthetic_multidim_dgp.py -q`
Expected: FAIL at import (module does not exist).

- [ ] **Step 3: Create the module with the helpers**

Create `src/skcausal/datasets/semi_synthetic_multidim.py`:

```python
"""Multi-dimensional semi-synthetic DGP driven by a real-outcome prognostic score."""

import numpy as np
import polars as pl
from scipy.special import log_ndtr

_LOG_2PI = np.log(2.0 * np.pi)


def _per_component(name, value, n_components, *, default=None):
    """Broadcast a scalar to every component or validate a length-k sequence."""

    if value is None:
        value = default
    if isinstance(value, str) or np.ndim(value) == 0:
        return [value] * n_components
    values = list(value)
    if len(values) != n_components:
        raise ValueError(
            f"{name} must be a scalar or a sequence of length "
            f"n_treatments={n_components}."
        )
    return values


def _validate_loadings(value, n_components):
    loadings = np.asarray(
        _per_component("confounding_loadings", value, n_components, default=1.0),
        dtype=float,
    )
    if loadings.shape != (n_components,) or not np.isfinite(loadings).all():
        raise ValueError("confounding_loadings must be finite.")
    return loadings


def _validate_levels(value, n_components):
    levels = _per_component("n_levels", value, n_components)
    for entry in levels:
        if entry is None:
            continue
        if isinstance(entry, bool) or not isinstance(entry, (int, np.integer)):
            raise TypeError("n_levels entries must be None or integers.")
        if entry < 2:
            raise ValueError("n_levels entries must be at least 2.")
    return [None if entry is None else int(entry) for entry in levels]


def _validate_targets(value, n_components):
    targets = _per_component("target_columns", value, n_components)
    for entry in targets:
        if entry is not None and not isinstance(entry, str):
            raise TypeError("target_columns entries must be None or column names.")
    named = [entry for entry in targets if entry is not None]
    if len(set(named)) != len(named):
        raise ValueError("target_columns must not repeat a column.")
    return targets


def _exchangeable_correlation(n_components, correlation):
    matrix = np.full((n_components, n_components), float(correlation))
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _interval_log_mass(lower, upper):
    """``log(Phi(upper) - Phi(lower))`` evaluated on the side with small mass."""

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        left = log_ndtr(upper) + np.log1p(-np.exp(log_ndtr(lower) - log_ndtr(upper)))
        right = log_ndtr(-lower) + np.log1p(
            -np.exp(log_ndtr(-upper) - log_ndtr(-lower))
        )
    return np.where(upper <= 0.0, left, right)


def _gaussian_logpdf(residual, cholesky):
    """Log-density of the trailing axis of ``residual`` under ``N(0, L L')``."""

    residual = np.asarray(residual, dtype=float)
    dimension = residual.shape[-1]
    if dimension == 0:
        return np.zeros(residual.shape[:-1])
    whitened = residual @ np.linalg.inv(cholesky).T
    log_determinant = float(np.log(np.diag(cholesky)).sum())
    return (
        -0.5 * np.sum(whitened**2, axis=-1)
        - log_determinant
        - 0.5 * dimension * _LOG_2PI
    )


def _conditional_gaussian(covariance, continuous, categorical):
    """Regression matrix and covariance of the categorical block given the continuous block."""

    continuous = np.asarray(continuous, dtype=int)
    categorical = np.asarray(categorical, dtype=int)
    if categorical.size == 0:
        return np.zeros((0, continuous.size)), np.zeros((0, 0))
    cov_dd = covariance[np.ix_(categorical, categorical)]
    if continuous.size == 0:
        return np.zeros((categorical.size, 0)), cov_dd
    cov_cc = covariance[np.ix_(continuous, continuous)]
    cov_dc = covariance[np.ix_(categorical, continuous)]
    regression = np.linalg.solve(cov_cc, cov_dc.T).T
    return regression, cov_dd - regression @ cov_dc.T


def _numeric_column(frame, name, label):
    series = frame.get_column(name)
    if not series.dtype.is_numeric():
        raise ValueError(f"The {label} must be numeric.")
    values = series.cast(pl.Float64).to_numpy().astype(float)
    finite = values[np.isfinite(values)]
    if np.unique(finite).size < 2:
        raise ValueError(f"The {label} must have at least two distinct finite values.")
    return values
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/datasets/test_semi_synthetic_multidim_dgp.py -q`
Expected: 7 PASS. `ruff check` and `ruff format --check` on both files: clean.

- [ ] **Step 5: Commit**

```bash
git add src/skcausal/datasets/semi_synthetic_multidim.py tests/datasets/test_semi_synthetic_multidim_dgp.py
git commit -m "feat(multidim): validation, exchangeable correlation, probit and Gaussian helpers"
```

---

### Task 4: `MultidimSemiSyntheticDataset` with continuous components

**Files:**
- Modify: `src/skcausal/datasets/semi_synthetic_multidim.py`
- Test: `tests/datasets/test_semi_synthetic_multidim_dgp.py`

**Interfaces:**
- Consumes: Task 1's `_split_source`, `_fit_causal_structure(..., fitted_confounders=, confounder_coefficients=)`, `score_map_.fitted_confounder_r2`; Task 2's `_gaussian_mixture_marginal_cdf`, `_bisect_marginal_quantiles`; Task 3's helpers.
- Produces: the public class and its fitted attributes: `treatment_columns_`, `n_levels_`, `target_columns_`, `confounding_loadings_`, `continuous_indices_`, `categorical_indices_`, `treatment_levels_` (dict index → labels), `feature_names_`, `feature_coefficients_`, `feature_assignment_matrix_`, `prognostic_fit_r2_`, `feature_fit_r2_`, `released_offsets_`, `released_scales_`, `treatment_only_coefficients_`, `noise_correlation_matrix_`, `noise_covariance_`, `latent_basis_bounds_`, `hinge_knots_`, `basis_means_`, `basis_scales_`, `reference_basis_`, `basis_sizes_`, `interaction_sizes_`, `outcome_main_effects_`, `outcome_modifier_coefficients_`, `interaction_coefficients_`, `confounding_strength_`, `confounding_bias_`, `confounding_bias_ratio_`, `cut_points_` (dict index → interior cut points), `latent_grid_bounds_`.
- Produces: private methods used by later tasks and tests: `_latent_means(scores, strength) -> (n, k)`, `_marginal_quantiles(column, probabilities)`, `_cut_points(means) -> dict`, `_log_density(latent_continuous, levels, means, cut_points)`, `_parse_treatment(treatment, *, allow_unknown) -> (continuous, levels, valid)`, `_release(latent) -> pl.DataFrame`, `_component_bases(latent_continuous, levels) -> (basis, features)`, `_interaction_effects(features, n_rows)`, `_bias_at_strength(strength) -> (bias, ratio)`, `_latent_grid_bounds(means)`.

This task ships the full class. Categorical and feature-mimic code paths are present but guarded by a temporary constructor check that Task 5 and Task 6 remove.

- [ ] **Step 1: Write the failing tests for continuous components**

Append to `tests/datasets/test_semi_synthetic_multidim_dgp.py` (add `MultidimSemiSyntheticDataset` to the module import and `from skcausal.datasets.kang_schafer import KangSchaferContinuous`):

```python
def _two_continuous(**overrides):
    parameters = {
        "n_treatments": 2,
        "treatment_correlation": 0.6,
        "randomized_weight": 0.3,
        "random_state": 5,
    }
    parameters.update(overrides)
    return MultidimSemiSyntheticDataset(ToyRealDataset(), **parameters)


def test_multidim_releases_named_continuous_columns_and_density_integrates():
    dataset = _two_continuous()
    X, treatment, outcome = dataset.load()

    assert list(X.columns) == ["age", "income", "region"]
    assert list(treatment.columns) == ["t_0", "t_1"]
    assert dataset.column_types == {"t_0": "continuous", "t_1": "continuous"}
    assert outcome.shape == (8, 1)
    assert 0.0 <= dataset.prognostic_fit_r2_ <= 1.0

    axis = np.linspace(-12.0, 12.0, 241)
    lattice = np.array(np.meshgrid(axis, axis, indexing="ij")).reshape(2, -1).T
    repeated_X = X.to_numpy()[np.zeros(lattice.shape[0], dtype=int)]
    density = np.exp(dataset.log_prob(repeated_X, lattice)[:, 0]).reshape(241, 241)
    integral = np.trapezoid(np.trapezoid(density, axis, axis=1), axis)
    np.testing.assert_allclose(integral, 1.0, atol=2e-3)


def test_multidim_log_prob_is_the_exact_gaussian_mixture():
    dataset = _two_continuous()
    X, treatment, _ = dataset.load()
    means = dataset._latent_means(dataset.source_scores_, dataset.confounding_strength_)
    values = treatment.to_numpy().astype(float)
    weight = dataset.randomized_weight
    expected = np.log(
        (1.0 - weight)
        * np.array(
            [
                multivariate_normal(mean=means[i], cov=dataset.noise_covariance_).pdf(values[i])
                for i in range(X.height)
            ]
        )
        + weight
        * multivariate_normal(mean=np.zeros(2), cov=dataset.noise_covariance_).pdf(values)
    )
    np.testing.assert_allclose(dataset.log_prob(X, treatment)[:, 0], expected, rtol=1e-10)
    assert np.isneginf(dataset.log_prob(X.head(1), np.array([[np.inf, 0.0]]))[0, 0])


def test_multidim_predict_y_matches_noise_free_outcome_and_reference():
    dataset = _two_continuous(outcome_noise_scale=0.0)
    X, treatment, outcome = dataset.load()
    np.testing.assert_allclose(dataset.predict_y(X, treatment), outcome.to_numpy())

    at_reference = dataset.predict_y(X, np.zeros((X.height, 2)))[:, 0]
    np.testing.assert_allclose(at_reference, dataset.source_mu0_)
    assert len(dataset.interaction_coefficients_) == 1
    assert dataset.interaction_coefficients_[(0, 1)].shape == (1, 1)


def test_multidim_zero_loading_component_is_independent_of_x():
    dataset = _two_continuous(confounding_loadings=[1.0, 0.0])
    means = dataset._latent_means(dataset.source_scores_, dataset.confounding_strength_)
    np.testing.assert_allclose(means[:, 1], 0.0)
    assert np.std(means[:, 0]) > 0.0


def test_multidim_sampled_correlation_tracks_rho():
    dataset = MultidimSemiSyntheticDataset(
        KangSchaferContinuous(n=4000, random_state=1),
        n_treatments=2,
        treatment_correlation=0.6,
        confounding_strength=0.0,
        randomized_weight=0.0,
        random_state=2,
    )
    _, treatment, _ = dataset.load()
    correlation = np.corrcoef(treatment.to_numpy().astype(float).T)[0, 1]
    np.testing.assert_allclose(correlation, 0.6, atol=0.05)


def test_multidim_zero_bias_when_randomized_or_unconfounded():
    randomized = _two_continuous(randomized_weight=1.0)
    assert randomized.confounding_bias_ == 0.0
    unconfounded = _two_continuous(confounding_strength=0.0)
    assert unconfounded.confounding_bias_ == 0.0
    assert unconfounded.confounding_bias_ratio_ == 0.0


def test_multidim_bias_increases_with_strength_and_calibrates():
    source = KangSchaferContinuous(n=300, random_state=2)
    biases = [
        MultidimSemiSyntheticDataset(
            source, n_treatments=2, confounding_strength=strength, random_state=3
        ).confounding_bias_ratio_
        for strength in (0.5, 1.0, 2.0)
    ]
    assert biases[0] < biases[1] < biases[2]

    calibrated = MultidimSemiSyntheticDataset(
        source, n_treatments=2, target_confounding_bias=0.5, random_state=3
    )
    np.testing.assert_allclose(calibrated.confounding_bias_ratio_, 0.5, atol=2e-3)
    assert calibrated.confounding_strength_ > 0.0


def test_multidim_grid_is_a_lattice_and_predict_curve_runs():
    dataset = _two_continuous()
    X, _, _ = dataset.load()
    grid = dataset.get_grid(25)
    assert grid.shape == (25, 2)
    assert grid.get_column("t_0").n_unique() == 5
    curve = np.asarray(dataset.predict(X, grid))
    assert curve.shape[0] == 25
    assert np.isfinite(curve).all()


def test_multidim_sample_is_reproducible_and_load_is_unchanged():
    dataset = _two_continuous()
    X, treatment, outcome = dataset.load()
    X_a, t_a, y_a = dataset.sample(random_state=11)
    X_b, t_b, y_b = dataset.sample(random_state=11)
    assert t_a.equals(t_b) and y_a.equals(y_b) and X_a.equals(X)
    assert not t_a.equals(treatment)
    X_again, treatment_again, outcome_again = dataset.load()
    assert treatment_again.equals(treatment) and outcome_again.equals(outcome)


def test_multidim_rejects_invalid_parameters():
    source = ToyRealDataset()
    with pytest.raises(ValueError, match="at least 1"):
        MultidimSemiSyntheticDataset(source, n_treatments=0)
    with pytest.raises(TypeError, match="integer"):
        MultidimSemiSyntheticDataset(source, n_treatments=2.0)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        MultidimSemiSyntheticDataset(source, treatment_correlation=1.0)
    with pytest.raises(ValueError, match="positive"):
        MultidimSemiSyntheticDataset(source, treatment_noise_scale=0.0)
    with pytest.raises(ValueError, match="treatment_interaction_scale"):
        MultidimSemiSyntheticDataset(source, treatment_interaction_scale=-1.0)
    with pytest.raises(ValueError, match="length n_treatments=2"):
        MultidimSemiSyntheticDataset(source, confounding_loadings=[1.0, 2.0, 3.0])


def test_multidim_requires_a_single_numeric_source_outcome():
    class TwoOutcomeDataset(ToyRealDataset):
        def _load(self):
            X, t, y = super()._load()
            return X, t, y.with_columns(pl.lit(1.0).alias("extra"))

    with pytest.raises(ValueError, match="exactly one outcome column"):
        MultidimSemiSyntheticDataset(TwoOutcomeDataset(), n_treatments=1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/datasets/test_semi_synthetic_multidim_dgp.py -q`
Expected: FAIL at import (`MultidimSemiSyntheticDataset` missing).

- [ ] **Step 3: Implement the class**

First replace the module header of `src/skcausal/datasets/semi_synthetic_multidim.py` (everything above `def _per_component`) with:

```python
"""Multi-dimensional semi-synthetic DGP driven by a real-outcome prognostic score."""

import itertools

import numpy as np
import polars as pl
from scipy.special import log_ndtr, logsumexp, ndtri

from skcausal.datasets.semi_synthetic import (
    BaseSemiSyntheticDataset,
    _bisect_marginal_quantiles,
    _gaussian_mixture_marginal_cdf,
    _log_mixture,
    _validate_nonnegative,
    _validate_shared_dgp_parameters,
)

__all__ = ["MultidimSemiSyntheticDataset"]

_LOG_2PI = np.log(2.0 * np.pi)
_BIAS_DESIGN_POINTS = 1024
_BIAS_CHUNK_ELEMENTS = 2**21
_MARGINAL_TAIL = 6.0
_BASIS_GRID_POINTS = 1001
```

Then append the class at the end of the module:

```python
class MultidimSemiSyntheticDataset(BaseSemiSyntheticDataset):
    r"""Semi-synthetic DGP with a vector of continuous and categorical treatments.

    A ridge prognostic score fitted on the real outcome ``y`` is the
    confounding backbone: it enters the baseline outcome with unit weight and
    the latent mean of component ``j`` with loading ``confounding_loadings[j]``
    times ``confounding_strength``. Components share exchangeable latent noise
    with correlation ``treatment_correlation``. A component may additionally
    mimic a numeric covariate named in ``target_columns``; that column is
    removed from the released ``X`` and its ridge prediction enters both the
    baseline outcome and that component's assignment. A component with
    ``n_levels`` set is released as ordered levels ``"0"`` to ``"K-1"`` by
    thresholding its latent at equal-mass marginal quantiles.

    Parameters
    ----------
    real_dataset : BaseDataset
        Source of covariates and of the outcome used for the prognostic score.
    n_treatments : int, default=2
        Number of treatment components, released as ``t_0`` to ``t_{k-1}``.
    confounding_loadings : float or sequence, default=None
        Per-component loading on the prognostic score; ``None`` means ``1.0``.
    n_levels : int, sequence, or None, default=None
        Per-component level count for discretized components; ``None`` keeps
        a component continuous.
    target_columns : str, sequence, or None, default=None
        Per-component numeric covariate to mimic; ``None`` mimics nothing.
    treatment_correlation : float, default=0.0
        Exchangeable latent noise correlation in ``[0, 1)``. Must be zero when
        two or more components are categorical.
    confounding_strength : float, default=1.0
        Nonnegative scale of the outcome-predictive assignment signal.
    target_confounding_bias : float or None, default=None
        Optional nonnegative normalized oracle-bias calibration target.
    treatment_only_strength : float, default=0.0
        Nonnegative scale of the residualized treatment-only predictors.
    randomized_weight : float, default=0.1
        Weight in ``[0, 1]`` of the covariate-independent mixture component.
    treatment_noise_scale : float, default=1.0
        Strictly positive latent noise standard deviation shared by components.
    outcome_effect_scale : float, default=1.0
        Nonnegative scale of treatment main-effect coefficients.
    effect_heterogeneity_scale : float, default=0.5
        Nonnegative scale of treatment-by-covariate coefficients.
    treatment_interaction_scale : float, default=0.5
        Nonnegative scale of cross-component effect coefficients.
    outcome_noise_scale : float, default=1.0
        Nonnegative Gaussian outcome-noise standard deviation.
    random_state : int, default=42
        Seed controlling frozen structure and the initial sample.
    """

    def __init__(
        self,
        real_dataset,
        n_treatments=2,
        confounding_loadings=None,
        n_levels=None,
        target_columns=None,
        treatment_correlation=0.0,
        confounding_strength=1.0,
        target_confounding_bias=None,
        treatment_only_strength=0.0,
        randomized_weight=0.1,
        treatment_noise_scale=1.0,
        outcome_effect_scale=1.0,
        effect_heterogeneity_scale=0.5,
        treatment_interaction_scale=0.5,
        outcome_noise_scale=1.0,
        random_state=42,
    ):
        if isinstance(n_treatments, bool) or not isinstance(
            n_treatments, (int, np.integer)
        ):
            raise TypeError("n_treatments must be an integer.")
        if n_treatments < 1:
            raise ValueError("n_treatments must be at least 1.")
        loadings = _validate_loadings(confounding_loadings, n_treatments)
        levels = _validate_levels(n_levels, n_treatments)
        targets = _validate_targets(target_columns, n_treatments)
        correlation = float(treatment_correlation)
        if not np.isfinite(correlation) or not 0.0 <= correlation < 1.0:
            raise ValueError("treatment_correlation must lie in [0, 1).")
        if correlation > 0.0 and sum(entry is not None for entry in levels) >= 2:
            raise ValueError(
                "treatment_correlation must be zero when two or more components "
                "are categorical."
            )
        if any(entry is not None for entry in levels) or any(
            entry is not None for entry in targets
        ):
            raise ValueError(
                "n_levels and target_columns are enabled in a later task."
            )
        if not np.isfinite(float(treatment_noise_scale)) or (
            float(treatment_noise_scale) <= 0.0
        ):
            raise ValueError("treatment_noise_scale must be positive.")
        _validate_nonnegative("treatment_interaction_scale", treatment_interaction_scale)
        _validate_shared_dgp_parameters(
            confounding_strength=confounding_strength,
            target_confounding_bias=target_confounding_bias,
            treatment_only_strength=treatment_only_strength,
            randomized_weight=randomized_weight,
            outcome_effect_scale=outcome_effect_scale,
            effect_heterogeneity_scale=effect_heterogeneity_scale,
            outcome_noise_scale=outcome_noise_scale,
        )
        self.n_treatments = n_treatments
        self.confounding_loadings = confounding_loadings
        self.n_levels = n_levels
        self.target_columns = target_columns
        self.treatment_correlation = treatment_correlation
        self.confounding_strength = confounding_strength
        self.target_confounding_bias = target_confounding_bias
        self.treatment_only_strength = treatment_only_strength
        self.randomized_weight = randomized_weight
        self.treatment_noise_scale = treatment_noise_scale
        self.outcome_effect_scale = outcome_effect_scale
        self.effect_heterogeneity_scale = effect_heterogeneity_scale
        self.treatment_interaction_scale = treatment_interaction_scale
        self.outcome_noise_scale = outcome_noise_scale
        self._validated_loadings = loadings
        self._validated_levels = levels
        self._validated_targets = targets
        super().__init__(real_dataset=real_dataset, random_state=random_state)

    # -- source split ----------------------------------------------------------

    def _split_source(self, X, treatment, outcome):
        outcome = self._coerce_backend_frame(outcome, backend="polars")
        if outcome.width != 1:
            raise ValueError("real_dataset must provide exactly one outcome column.")
        self._source_outcome = _numeric_column(
            outcome, outcome.columns[0], "source outcome"
        )
        feature_names = [name for name in self._validated_targets if name is not None]
        self._source_features = {}
        for name in feature_names:
            if name not in X.columns:
                raise ValueError(
                    f"target column {name!r} is not a covariate of real_dataset."
                )
            self._source_features[name] = _numeric_column(
                X, name, f"target column {name!r}"
            )
        released = X.drop(feature_names)
        if released.width == 0:
            raise ValueError(
                "At least one covariate must remain after removing target_columns."
            )
        return released

    # -- structural preparation ------------------------------------------------

    def _prepare_dgp(self, X, rng):
        n_components = int(self.n_treatments)
        self.treatment_columns_ = [f"t_{index}" for index in range(n_components)]
        self.n_levels_ = list(self._validated_levels)
        self.target_columns_ = list(self._validated_targets)
        self.confounding_loadings_ = self._validated_loadings.copy()
        self.continuous_indices_ = np.array(
            [j for j, levels in enumerate(self.n_levels_) if levels is None], dtype=int
        )
        self.categorical_indices_ = np.array(
            [j for j, levels in enumerate(self.n_levels_) if levels is not None],
            dtype=int,
        )
        self.column_types = {
            name: "continuous" if levels is None else "categorical"
            for name, levels in zip(self.treatment_columns_, self.n_levels_)
        }
        self.treatment_levels_ = {
            int(j): [str(level) for level in range(self.n_levels_[j])]
            for j in self.categorical_indices_
        }

        self.feature_names_ = [
            name for name in self.target_columns_ if name is not None
        ]
        self.feature_coefficients_ = rng.normal(size=len(self.feature_names_))
        fitted = np.column_stack(
            [self._source_outcome]
            + [self._source_features[name] for name in self.feature_names_]
        )
        scores = self._fit_causal_structure(
            X,
            rng,
            fitted_confounders=fitted,
            confounder_coefficients=np.concatenate(
                ([1.0], self.feature_coefficients_)
            ),
        )
        r2 = self.score_map_.fitted_confounder_r2
        self.prognostic_fit_r2_ = float(r2[0])
        self.feature_fit_r2_ = dict(zip(self.feature_names_, r2[1:].tolist()))
        self.feature_assignment_matrix_ = np.zeros(
            (len(self.feature_names_), n_components)
        )
        self.released_offsets_ = np.zeros(n_components)
        self.released_scales_ = np.ones(n_components)
        for j, name in enumerate(self.target_columns_):
            if name is None:
                continue
            position = self.feature_names_.index(name)
            self.feature_assignment_matrix_[position, j] = self.feature_coefficients_[
                position
            ]
            if self.n_levels_[j] is None:
                values = self._source_features[name]
                finite = values[np.isfinite(values)]
                self.released_offsets_[j] = float(finite.mean())
                self.released_scales_[j] = float(finite.std(ddof=0))

        self.treatment_only_coefficients_ = rng.normal(
            scale=1.0 / np.sqrt(2.0), size=(2, n_components)
        )
        self.noise_correlation_matrix_ = _exchangeable_correlation(
            n_components, self.treatment_correlation
        )
        self.noise_covariance_ = (
            float(self.treatment_noise_scale) ** 2 * self.noise_correlation_matrix_
        )
        self._noise_cholesky = np.linalg.cholesky(self.noise_covariance_)
        continuous, categorical = self.continuous_indices_, self.categorical_indices_
        self._continuous_cholesky = (
            np.linalg.cholesky(self.noise_covariance_[np.ix_(continuous, continuous)])
            if continuous.size
            else np.zeros((0, 0))
        )
        self._conditional_regression, conditional_covariance = _conditional_gaussian(
            self.noise_covariance_, continuous, categorical
        )
        self._conditional_scales = np.sqrt(np.diag(conditional_covariance))

        self.latent_basis_bounds_ = ndtri(np.array([0.01, 0.99]))
        self.hinge_knots_ = ndtri(np.array([1.0 / 3.0, 2.0 / 3.0]))
        raw = self._raw_continuous_basis(
            np.linspace(*self.latent_basis_bounds_, _BASIS_GRID_POINTS)
        )
        self.basis_means_ = raw.mean(axis=0)
        self.basis_scales_ = raw.std(axis=0, ddof=0)
        self.basis_scales_[self.basis_scales_ == 0.0] = 1.0
        self.reference_basis_ = (
            (self._raw_continuous_basis(np.zeros(1)) - self.basis_means_)
            / self.basis_scales_
        )[0]
        self.basis_sizes_ = [
            5 if levels is None else levels - 1 for levels in self.n_levels_
        ]
        self.interaction_sizes_ = [
            1 if levels is None else levels - 1 for levels in self.n_levels_
        ]
        total = int(sum(self.basis_sizes_))
        self.outcome_main_effects_ = rng.normal(
            scale=self.outcome_effect_scale, size=total
        )
        self.outcome_modifier_coefficients_ = rng.normal(
            scale=self.effect_heterogeneity_scale, size=(total, 2)
        )
        self.interaction_coefficients_ = {
            (j, l): rng.normal(
                scale=self.treatment_interaction_scale,
                size=(self.interaction_sizes_[j], self.interaction_sizes_[l]),
            )
            for j in range(n_components)
            for l in range(j + 1, n_components)
        }

        self._bias_rows = rng.integers(self.n, size=_BIAS_DESIGN_POINTS)
        self._bias_noise = (
            rng.standard_normal((_BIAS_DESIGN_POINTS, n_components))
            @ self._noise_cholesky.T
        )
        self._bias_uniforms = rng.random(_BIAS_DESIGN_POINTS)
        self._level_combinations = np.array(
            list(itertools.product(*[range(self.n_levels_[j]) for j in categorical])),
            dtype=int,
        ).reshape(-1, categorical.size)

        self._resolve_confounding_strength(
            self._bias_at_strength,
            confounding_strength=self.confounding_strength,
            target_confounding_bias=self.target_confounding_bias,
            randomized_weight=self.randomized_weight,
        )
        means = self._latent_means(scores, self.confounding_strength_)
        self.cut_points_ = self._cut_points(means)
        self.latent_grid_bounds_ = self._latent_grid_bounds(means)

    # -- assignment law --------------------------------------------------------

    def _latent_means(self, scores, strength):
        confounders = scores.confounders
        signal = confounders[:, :1] * self.confounding_loadings_[None, :]
        if self.feature_names_:
            signal = signal + confounders[:, 1:] @ self.feature_assignment_matrix_
        treatment_only = scores.treatment_only @ self.treatment_only_coefficients_
        means = strength * signal + self.treatment_only_strength * treatment_only
        if not np.isfinite(means).all():
            raise ValueError("Derived latent treatment means must be finite.")
        return means

    def _marginal_quantiles(self, column, probabilities):
        scale = float(self.treatment_noise_scale)
        lower = min(float(column.min()), 0.0) - _MARGINAL_TAIL * scale
        upper = max(float(column.max()), 0.0) + _MARGINAL_TAIL * scale
        return _bisect_marginal_quantiles(
            lambda value: _gaussian_mixture_marginal_cdf(
                value, column, scale, self.randomized_weight
            ),
            probabilities,
            lower,
            upper,
        )

    def _cut_points(self, means):
        return {
            int(j): self._marginal_quantiles(
                means[:, j], np.arange(1, self.n_levels_[j]) / self.n_levels_[j]
            )
            for j in self.categorical_indices_
        }

    def _log_density(self, latent_continuous, levels, means, cut_points):
        """Joint mixture log-density of continuous latents and categorical levels.

        Leading dimensions of ``latent_continuous`` ``(..., k_C)``, ``levels``
        ``(..., k_D)`` and ``means`` ``(..., k)`` broadcast against each other.
        """

        continuous, categorical = self.continuous_indices_, self.categorical_indices_

        def component(location):
            residual = latent_continuous - location[..., continuous]
            value = _gaussian_logpdf(residual, self._continuous_cholesky)
            if categorical.size:
                conditional = (
                    location[..., categorical]
                    + residual @ self._conditional_regression.T
                )
                for position, j in enumerate(categorical):
                    bounds = np.concatenate(([-np.inf], cut_points[int(j)], [np.inf]))
                    index = levels[..., position]
                    scale = self._conditional_scales[position]
                    value = value + _interval_log_mass(
                        (bounds[index] - conditional[..., position]) / scale,
                        (bounds[index + 1] - conditional[..., position]) / scale,
                    )
            return value

        return _log_mixture(
            component(means), component(np.zeros_like(means)), self.randomized_weight
        )

    def _release(self, latent):
        columns = {}
        for j, name in enumerate(self.treatment_columns_):
            if self.n_levels_[j] is None:
                columns[name] = (
                    self.released_offsets_[j] + self.released_scales_[j] * latent[:, j]
                )
            else:
                index = np.searchsorted(self.cut_points_[j], latent[:, j], side="left")
                columns[name] = np.asarray(self.treatment_levels_[j], dtype=str)[index]
        return pl.DataFrame(columns)

    def _sample_treatment(self, X, rng):
        means = self._latent_means(self._scores(X), self.confounding_strength_)
        randomized = rng.random(means.shape[0]) < self.randomized_weight
        location = np.where(randomized[:, None], 0.0, means)
        latent = location + rng.standard_normal(means.shape) @ self._noise_cholesky.T
        return self._release(latent)

    def _parse_treatment(self, treatment, *, allow_unknown):
        """Return standardized continuous latents, level indices, and a validity mask."""

        n_rows = treatment.height
        continuous = np.empty((n_rows, self.continuous_indices_.size))
        valid = np.ones(n_rows, dtype=bool)
        for position, j in enumerate(self.continuous_indices_):
            values = (
                treatment.get_column(self.treatment_columns_[j])
                .cast(pl.Float64)
                .to_numpy()
                .astype(float)
            )
            valid &= np.isfinite(values)
            continuous[:, position] = (
                values - self.released_offsets_[j]
            ) / self.released_scales_[j]
        levels = np.empty((n_rows, self.categorical_indices_.size), dtype=int)
        for position, j in enumerate(self.categorical_indices_):
            labels = treatment.get_column(self.treatment_columns_[j]).cast(pl.Utf8)
            index = (
                labels.replace_strict(
                    {
                        level: index
                        for index, level in enumerate(self.treatment_levels_[j])
                    },
                    default=-1,
                    return_dtype=pl.Int64,
                )
                .fill_null(-1)
                .to_numpy()
            )
            levels[:, position] = index
            valid &= index >= 0
        if not allow_unknown and not valid.all():
            raise ValueError(
                "Treatment values must be finite continuous values and known "
                "categorical levels."
            )
        return continuous, levels, valid

    def _log_prob(self, X, treatment):
        continuous, levels, valid = self._parse_treatment(treatment, allow_unknown=True)
        log_density = np.full(treatment.height, -np.inf)
        if not valid.any():
            return log_density
        means = self._latent_means(self._scores(X), self.confounding_strength_)
        jacobian = -float(
            np.log(self.released_scales_[self.continuous_indices_]).sum()
        )
        log_density[valid] = (
            self._log_density(
                continuous[valid], levels[valid], means[valid], self.cut_points_
            )
            + jacobian
        )
        return log_density

    # -- outcome surface -------------------------------------------------------

    def _raw_continuous_basis(self, standardized):
        v = np.asarray(standardized, dtype=float).reshape(-1)
        return np.column_stack(
            (
                v,
                -np.expm1(-0.5 * v**2),
                np.sin(v),
                np.maximum(v - self.hinge_knots_[0], 0.0),
                np.maximum(v - self.hinge_knots_[1], 0.0),
            )
        )

    def _component_bases(self, latent_continuous, levels):
        """Stacked outcome basis ``(n, P)`` and per-component cross-term features."""

        scale = float(self.treatment_noise_scale)
        bases, features = [], []
        continuous_position = categorical_position = 0
        for j in range(int(self.n_treatments)):
            if self.n_levels_[j] is None:
                v = latent_continuous[:, continuous_position] / scale
                continuous_position += 1
                bases.append(
                    (self._raw_continuous_basis(v) - self.basis_means_)
                    / self.basis_scales_
                    - self.reference_basis_
                )
                features.append(np.tanh(v)[:, None])
            else:
                index = levels[:, categorical_position]
                categorical_position += 1
                indicators = (
                    index[:, None] == np.arange(1, self.n_levels_[j])[None, :]
                ).astype(float)
                bases.append(indicators)
                features.append(indicators)
        return np.column_stack(bases), features

    def _interaction_effects(self, features, n_rows):
        total = np.zeros(n_rows)
        for (j, l), coefficients in self.interaction_coefficients_.items():
            total = total + np.einsum(
                "ip,pq,iq->i", features[j], coefficients, features[l]
            )
        return total

    def _predict_y(self, X, treatment):
        continuous, levels, _ = self._parse_treatment(treatment, allow_unknown=False)
        basis, features = self._component_bases(continuous, levels)
        mean = self._outcome_mean(
            self._scores(X),
            basis,
            self.outcome_main_effects_,
            self.outcome_modifier_coefficients_,
        )
        return mean + self._interaction_effects(features, continuous.shape[0])

    def _sample_outcome(self, mean, X, treatment, rng):
        return self._gaussian_outcome(mean, self.outcome_noise_scale, rng)

    # -- confounding bias ------------------------------------------------------

    def _bias_at_strength(self, strength):
        if self.randomized_weight == 1.0 or (
            strength == 0.0 and self.treatment_only_strength == 0.0
        ):
            return 0.0, 0.0

        scores = self.source_scores_
        means = self._latent_means(scores, strength)
        cut_points = self._cut_points(means)
        location = np.where(
            self._bias_uniforms[:, None] < self.randomized_weight,
            0.0,
            means[self._bias_rows],
        )
        latent = (location + self._bias_noise)[:, self.continuous_indices_]
        mu0 = self.source_mu0_
        modifiers = scores.modifiers
        n_rows = means.shape[0]
        n_points = latent.shape[0]
        combinations = self._level_combinations
        marginal = np.empty((n_points, len(combinations)))
        distortion = np.empty((n_points, len(combinations)))
        chunk = max(1, _BIAS_CHUNK_ELEMENTS // n_rows)
        for combo_index, combination in enumerate(combinations):
            levels = np.broadcast_to(combination, (n_points, combination.size))
            basis, _ = self._component_bases(latent, levels)
            modifier_effects = basis @ self.outcome_modifier_coefficients_
            for start in range(0, n_points, chunk):
                stop = min(start + chunk, n_points)
                log_density = self._log_density(
                    latent[start:stop, None, :],
                    levels[start:stop, None, :],
                    means[None, :, :],
                    cut_points,
                )
                log_total = logsumexp(log_density, axis=1)
                supported = np.isfinite(log_total)
                weights = np.zeros_like(log_density)
                weights[supported] = np.exp(
                    log_density[supported] - log_total[supported, None]
                )
                marginal[start:stop, combo_index] = np.where(
                    supported, np.exp(log_total - np.log(n_rows)), 0.0
                )
                weighted_mu0 = np.where(supported, weights @ mu0, mu0.mean())
                weighted_modifiers = np.where(
                    supported[:, None], weights @ modifiers, modifiers.mean(axis=0)
                )
                distortion[start:stop, combo_index] = (
                    weighted_mu0 - mu0.mean()
                ) + np.einsum(
                    "ij,ij->i",
                    modifier_effects[start:stop],
                    weighted_modifiers - modifiers.mean(axis=0),
                )
        total = marginal.sum(axis=1, keepdims=True)
        if not np.isfinite(total).all() or (total <= 0.0).any():
            raise ValueError(
                "Multidimensional confounding bias is not representable at this "
                "configuration: a design point has no finite assignment density."
            )
        probabilities = marginal / total
        bias = float(np.sqrt(np.mean(np.sum(probabilities * distortion**2, axis=1))))
        if not np.isfinite(bias):
            raise ValueError("Multidimensional confounding bias must be finite.")
        return bias, bias / self.baseline_standard_deviation_

    # -- grid ------------------------------------------------------------------

    def _latent_grid_bounds(self, means):
        central = self.latent_basis_bounds_ * float(self.treatment_noise_scale)
        bounds = np.empty((self.continuous_indices_.size, 2))
        for position, j in enumerate(self.continuous_indices_):
            quantiles = self._marginal_quantiles(means[:, j], np.array([0.01, 0.99]))
            bounds[position] = (
                min(quantiles[0], central[0]),
                max(quantiles[1], central[1]),
            )
        return bounds

    def _get_grid(self, n):
        n_continuous = self.continuous_indices_.size
        points = max(2, int(round(n ** (1.0 / n_continuous)))) if n_continuous else 0
        grid = None
        for j, name in enumerate(self.treatment_columns_):
            if self.n_levels_[j] is None:
                position = int(np.flatnonzero(self.continuous_indices_ == j)[0])
                latent = np.linspace(*self.latent_grid_bounds_[position], points)
                frame = pl.DataFrame(
                    {name: self.released_offsets_[j] + self.released_scales_[j] * latent}
                )
            else:
                frame = pl.DataFrame({name: self.treatment_levels_[j]})
            grid = frame if grid is None else grid.join(frame, how="cross")
        return grid

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from skcausal.datasets.kang_schafer import KangSchaferContinuous

        return [
            {
                "real_dataset": KangSchaferContinuous(n=64, random_state=4),
                "random_state": 7,
            }
        ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/datasets/test_semi_synthetic_multidim_dgp.py -v`
Expected: all PASS. If `test_multidim_bias_increases_with_strength_and_calibrates` is slow (more than ~20 s), reduce `KangSchaferContinuous(n=300)` to `n=200`; do not change `_BIAS_DESIGN_POINTS`.

Then: `ruff check src/skcausal/datasets/semi_synthetic_multidim.py tests/datasets/test_semi_synthetic_multidim_dgp.py && ruff format --check` on both. Fix formatting with `ruff format` on those two files if needed.

- [ ] **Step 5: Commit**

```bash
git add src/skcausal/datasets/semi_synthetic_multidim.py tests/datasets/test_semi_synthetic_multidim_dgp.py
git commit -m "feat(multidim): vector continuous semi-synthetic DGP with exact oracles and MC bias metric"
```

---

### Task 5: Categorical components via ordered-probit discretization

**Files:**
- Modify: `src/skcausal/datasets/semi_synthetic_multidim.py` (constructor guard)
- Test: `tests/datasets/test_semi_synthetic_multidim_dgp.py`

**Interfaces:**
- Consumes: Task 4's `_cut_points`, `_release`, `_parse_treatment`, `_log_density`, `_level_combinations`, `_get_grid`.
- Produces: `n_levels` accepted by the constructor; `cut_points_[j]` is a `(K - 1,)` array for each categorical component; categorical columns released as strings with `column_types[name] == "categorical"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/datasets/test_semi_synthetic_multidim_dgp.py`:

```python
def _mixed(**overrides):
    parameters = {
        "n_treatments": 2,
        "n_levels": [None, 3],
        "treatment_correlation": 0.6,
        "randomized_weight": 0.3,
        "random_state": 9,
    }
    parameters.update(overrides)
    return MultidimSemiSyntheticDataset(ToyRealDataset(), **parameters)


def test_mixed_masses_sum_to_the_continuous_marginal_and_match_brute_force():
    dataset = _mixed()
    X, treatment, _ = dataset.load()
    assert dataset.column_types == {"t_0": "continuous", "t_1": "categorical"}
    assert treatment.get_column("t_1").cast(pl.Utf8).is_in(["0", "1", "2"]).all()
    assert dataset.cut_points_[1].shape == (2,)

    row = X.head(1)
    means = dataset._latent_means(dataset.source_scores_, dataset.confounding_strength_)[0]
    weight = dataset.randomized_weight
    covariance = dataset.noise_covariance_
    value = 0.7
    masses = np.array(
        [
            np.exp(dataset.log_prob(row, pl.DataFrame({"t_0": [value], "t_1": [level]}))[0, 0])
            for level in ("0", "1", "2")
        ]
    )
    marginal = (1.0 - weight) * multivariate_normal(mean=means[:1], cov=covariance[:1, :1]).pdf(
        value
    ) + weight * multivariate_normal(mean=[0.0], cov=covariance[:1, :1]).pdf(value)
    np.testing.assert_allclose(masses.sum(), marginal, rtol=1e-10)

    bounds = np.concatenate(([-12.0], dataset.cut_points_[1], [12.0]))
    for level in range(3):
        z1 = np.linspace(bounds[level], bounds[level + 1], 20001)
        points = np.column_stack((np.full(z1.size, value), z1))
        joint = (1.0 - weight) * multivariate_normal(mean=means, cov=covariance).pdf(
            points
        ) + weight * multivariate_normal(mean=np.zeros(2), cov=covariance).pdf(points)
        np.testing.assert_allclose(masses[level], np.trapezoid(joint, z1), rtol=1e-4)


def test_categorical_levels_are_balanced_and_grid_crosses_levels():
    dataset = MultidimSemiSyntheticDataset(
        KangSchaferContinuous(n=4000, random_state=1),
        n_treatments=2,
        n_levels=[None, 4],
        confounding_strength=0.0,
        randomized_weight=0.0,
        random_state=2,
    )
    _, treatment, _ = dataset.load()
    counts = treatment.get_column("t_1").cast(pl.Utf8).value_counts().sort("t_1")
    assert counts.get_column("t_1").to_list() == ["0", "1", "2", "3"]
    assert np.all(np.abs(counts.get_column("count").to_numpy() - 1000) < 120)

    grid = dataset.get_grid(10)
    assert grid.shape == (40, 2)
    assert grid.get_column("t_0").n_unique() == 10
    assert grid.get_column("t_1").cast(pl.Utf8).n_unique() == 4


def test_categorical_only_multidim_has_level_lattice_and_exact_calibration():
    dataset = MultidimSemiSyntheticDataset(
        KangSchaferContinuous(n=300, random_state=2),
        n_treatments=2,
        n_levels=[2, 3],
        target_confounding_bias=0.3,
        random_state=4,
    )
    X, _, _ = dataset.load()
    grid = dataset.get_grid(100)
    assert grid.shape == (6, 2)

    repeated_X = X.head(1).to_numpy()[np.zeros(6, dtype=int)]
    masses = np.exp(dataset.log_prob(repeated_X, grid)[:, 0])
    np.testing.assert_allclose(masses.sum(), 1.0, rtol=1e-10)
    np.testing.assert_allclose(dataset.confounding_bias_ratio_, 0.3, atol=2e-3)
    assert dataset.interaction_coefficients_[(0, 1)].shape == (1, 2)


def test_unknown_level_is_unsupported_and_predict_y_rejects_it():
    dataset = _mixed()
    X, _, _ = dataset.load()
    bad = pl.DataFrame({"t_0": [0.0], "t_1": ["7"]})
    assert np.isneginf(dataset.log_prob(X.head(1), bad)[0, 0])
    with pytest.raises(ValueError, match="known categorical levels"):
        dataset.predict_y(X.head(1), bad)


def test_correlated_noise_with_two_categorical_components_is_rejected():
    with pytest.raises(ValueError, match="two or more components are categorical"):
        MultidimSemiSyntheticDataset(
            ToyRealDataset(), n_treatments=2, n_levels=[2, 3], treatment_correlation=0.2
        )
    MultidimSemiSyntheticDataset(
        ToyRealDataset(), n_treatments=2, n_levels=[2, 3], treatment_correlation=0.0
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/datasets/test_semi_synthetic_multidim_dgp.py -q -k "mixed or categorical or unknown_level or correlated_noise"`
Expected: FAIL with `n_levels and target_columns are enabled in a later task`.

- [ ] **Step 3: Remove the guard for `n_levels`**

In the constructor, replace the temporary guard with one that only guards `target_columns`:

```python
        if any(entry is not None for entry in targets):
            raise ValueError("target_columns is enabled in a later task.")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/datasets/test_semi_synthetic_multidim_dgp.py -v`
Expected: all PASS. If a categorical test fails, the likely culprits are: `enforce_dtypes` returning a `Categorical`/`Enum` column that `.cast(pl.Utf8)` handles (it does), the `searchsorted` side (`"left"` gives level `k` for `tau_{k-1} < z <= tau_k`), or the `bounds[index + 1]` lookup when `index` is a broadcast `(m, 1)` array (it must stay integer dtype). Fix the implementation, not the test.

Then `ruff check` and `ruff format --check` on both files.

- [ ] **Step 5: Commit**

```bash
git add src/skcausal/datasets/semi_synthetic_multidim.py tests/datasets/test_semi_synthetic_multidim_dgp.py
git commit -m "feat(multidim): ordered-probit categorical components with exact mixed law"
```

---

### Task 6: Feature mimicking via `target_columns`

**Files:**
- Modify: `src/skcausal/datasets/semi_synthetic_multidim.py` (constructor guard, `get_test_params`)
- Test: `tests/datasets/test_semi_synthetic_multidim_dgp.py`

**Interfaces:**
- Consumes: Task 4's `_split_source`, `feature_assignment_matrix_`, `released_offsets_`, `released_scales_`, `feature_fit_r2_`.
- Produces: `target_columns` accepted by the constructor; a second `get_test_params` entry exercising a mixed configuration with a mimicked feature and correlated noise.

- [ ] **Step 1: Write the failing tests**

Append to `tests/datasets/test_semi_synthetic_multidim_dgp.py`:

```python
def test_feature_target_is_removed_from_x_and_released_on_its_scale():
    source = ToyRealDataset()
    dataset = MultidimSemiSyntheticDataset(
        source, n_treatments=2, target_columns=[None, "income"], random_state=3
    )
    X, treatment, _ = dataset.load()
    income = source.load()[0].get_column("income").to_numpy().astype(float)

    assert list(X.columns) == ["age", "region"]
    assert list(source.load()[0].columns) == ["age", "income", "region"]
    assert dataset.target_columns_ == [None, "income"]
    assert dataset.feature_names_ == ["income"]
    assert set(dataset.feature_fit_r2_) == {"income"}
    np.testing.assert_allclose(dataset.released_offsets_, [0.0, income.mean()])
    np.testing.assert_allclose(dataset.released_scales_, [1.0, income.std(ddof=0)])
    assert dataset.feature_assignment_matrix_.shape == (1, 2)
    assert dataset.feature_assignment_matrix_[0, 0] == 0.0
    assert dataset.feature_assignment_matrix_[0, 1] == dataset.feature_coefficients_[0]

    # Oracles accept the reduced schema, including NumPy of the reduced width.
    assert dataset.log_prob(X.to_numpy(), treatment).shape == (8, 1)
    with pytest.raises(ValueError):
        dataset.log_prob(source.load()[0], treatment)

    # Released value = mean + sd * latent, so log_prob at the feature mean equals
    # the latent density at zero minus log(sd).
    at_mean = np.zeros((8, 2))
    at_mean[:, 1] = income.mean()
    means = dataset._latent_means(dataset.source_scores_, dataset.confounding_strength_)
    latent_density = dataset._log_density(
        np.zeros((8, 2)), np.zeros((8, 0), dtype=int), means, dataset.cut_points_
    )
    np.testing.assert_allclose(
        dataset.log_prob(X, at_mean)[:, 0],
        latent_density - np.log(income.std(ddof=0)),
    )


def test_feature_score_enters_assignment_and_baseline():
    dataset = MultidimSemiSyntheticDataset(
        ToyRealDataset(),
        n_treatments=2,
        confounding_loadings=[1.0, 0.0],
        target_columns=[None, "income"],
        random_state=6,
    )
    means = dataset._latent_means(dataset.source_scores_, dataset.confounding_strength_)
    assert np.std(means[:, 1]) > 0.0
    assert dataset.score_map_.fitted_confounder_r2.shape == (2,)
    np.testing.assert_allclose(
        dataset.baseline_surface_.confounder_coefficients,
        [1.0, dataset.feature_coefficients_[0]],
    )


def test_feature_target_validation():
    source = ToyRealDataset()
    with pytest.raises(ValueError, match="not a covariate"):
        MultidimSemiSyntheticDataset(source, n_treatments=1, target_columns="height")
    with pytest.raises(ValueError, match="must be numeric"):
        MultidimSemiSyntheticDataset(source, n_treatments=1, target_columns="region")
    with pytest.raises(ValueError, match="At least one covariate"):
        MultidimSemiSyntheticDataset(
            KangSchaferContinuous(n=50, random_state=0),
            n_treatments=4,
            target_columns=["x1", "x2", "x3", "x4"],
        )
    mixed = MultidimSemiSyntheticDataset(
        source, n_treatments=2, n_levels=[None, 2], target_columns=["age", "income"], random_state=1
    )
    X, treatment, _ = mixed.load()
    assert list(X.columns) == ["region"]
    assert mixed.released_scales_[1] == 1.0
    assert treatment.get_column("t_1").cast(pl.Utf8).is_in(["0", "1"]).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/datasets/test_semi_synthetic_multidim_dgp.py -q -k feature`
Expected: FAIL with `target_columns is enabled in a later task`.

- [ ] **Step 3: Remove the guard and extend `get_test_params`**

Delete the remaining temporary guard from the constructor. Replace `get_test_params`:

```python
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from skcausal.datasets.kang_schafer import KangSchaferContinuous

        return [
            {
                "real_dataset": KangSchaferContinuous(n=64, random_state=4),
                "random_state": 7,
            },
            {
                "real_dataset": KangSchaferContinuous(n=64, random_state=4),
                "n_treatments": 3,
                "confounding_loadings": [1.0, 0.5, 0.0],
                "n_levels": [None, None, 3],
                "target_columns": [None, "x2", None],
                "treatment_correlation": 0.4,
                "random_state": 8,
            },
        ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/datasets/test_semi_synthetic_multidim_dgp.py -v`
Expected: all PASS. Also confirm both test-parameter sets construct:

```bash
cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -c "
from skcausal.datasets.semi_synthetic_multidim import MultidimSemiSyntheticDataset as D
for p in D.get_test_params():
    d = D(**p); X, t, y = d.load(); print(X.shape, t.shape, dict(d.column_types), round(d.confounding_bias_ratio_, 3))
"
```

Then `ruff check` and `ruff format --check` on both files.

- [ ] **Step 5: Commit**

```bash
git add src/skcausal/datasets/semi_synthetic_multidim.py tests/datasets/test_semi_synthetic_multidim_dgp.py
git commit -m "feat(multidim): mimic numeric covariates as treatment components"
```

---

### Task 7: Export, discovery, documentation, full suite

**Files:**
- Modify: `src/skcausal/datasets/__init__.py`
- Modify: `tests/test_lookup.py`
- Modify: `docs/research/semi_synthetic_datasets.qmd` (after Step 6, before "## Configuration reference"; and the reference table)

**Interfaces:**
- Consumes: the finished `MultidimSemiSyntheticDataset`.
- Produces: `from skcausal.datasets import MultidimSemiSyntheticDataset`; discovery through `all_datasets`.

- [ ] **Step 1: Write the failing discovery test**

In `tests/test_lookup.py`, add `MultidimSemiSyntheticDataset` to the `skcausal.datasets` import and extend the discovery test:

```python
    assert MultidimSemiSyntheticDataset.__name__ == "MultidimSemiSyntheticDataset"
    assert "MultidimSemiSyntheticDataset" in names
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/test_lookup.py -q`
Expected: FAIL at import (`MultidimSemiSyntheticDataset` not exported).

- [ ] **Step 3: Export the class**

In `src/skcausal/datasets/__init__.py`, add after the continuous import:

```python
from .semi_synthetic_multidim import MultidimSemiSyntheticDataset
```

and add `"MultidimSemiSyntheticDataset",` to `__all__` between `"ModelInducedConfoundingRegressor"` and `"NurseStaffing"`.

- [ ] **Step 4: Run the discovery test**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests/test_lookup.py -q`
Expected: PASS.

- [ ] **Step 5: Document the new DGP**

In `docs/research/semi_synthetic_datasets.qmd`, insert before `## Configuration reference`:

````markdown
## Step 7: Multi-dimensional and mixed treatments

`MultidimSemiSyntheticDataset` releases a vector of treatments. A ridge
prognostic score fitted on the real outcome is the shared confounder; each
component loads on it with its own weight, so one dataset can hold strongly,
mildly, and un-confounded components. `treatment_correlation` adds latent noise
correlation that covariates do not explain. A component can mimic a numeric
covariate, which is then removed from `X`, and any component can be released
as ordered levels by thresholding its latent at equal-mass quantiles. The joint
`log_prob` stays exact, and `target_confounding_bias` calibrates one strength
for the whole vector.

```{python}
from skcausal.datasets import MultidimSemiSyntheticDataset

target = next(
    name
    for name in X_source.columns
    if X_source.get_column(name).dtype.is_numeric()
    and X_source.get_column(name).n_unique() > 2
)
multidim = MultidimSemiSyntheticDataset(
    source,
    n_treatments=3,
    confounding_loadings=[1.0, 0.5, 0.0],
    n_levels=[None, None, 3],
    target_columns=[None, target, None],
    treatment_correlation=0.4,
    target_confounding_bias=0.5,
    random_state=0,
)
X, treatment, outcome = multidim.load()
print(treatment.head())
print(
    {
        "mimicked column": target,
        "prognostic R2": round(multidim.prognostic_fit_r2_, 3),
        "feature R2": {k: round(v, 3) for k, v in multidim.feature_fit_r2_.items()},
        "bias ratio": round(multidim.confounding_bias_ratio_, 3),
    }
)
np.corrcoef(treatment.select(["t_0", "t_1"]).to_numpy().astype(float).T)[0, 1]
```

`get_grid` returns a lattice over the continuous components crossed with the
levels of the categorical ones, so `predict` evaluates the average response
surface over the whole vector.
````

Add these rows to the configuration reference table (the "Applies to" column uses `multidim`):

```markdown
| `n_treatments`, `confounding_loadings` | multidim | Number of components and each component's loading on the prognostic score. |
| `n_levels`, `target_columns` | multidim | Per-component discretization into ordered levels and the numeric covariate a component mimics (removed from `X`). |
| `treatment_correlation` | multidim | Exchangeable latent noise correlation in `[0, 1)`; zero when two or more components are categorical. |
| `treatment_interaction_scale` | multidim | Size of bounded cross-component effects. |
```

Also update the `treatment_noise_scale` row's "Applies to" cell to `continuous, multidim`.

- [ ] **Step 6: Render check of the notebook chunk**

Run the new chunk's code outside Quarto to make sure it executes on IHDP within a reasonable time:

```bash
cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src timeout 600 python -c "
import numpy as np
from skcausal.datasets import IHDPContinuous, MultidimSemiSyntheticDataset
source = IHDPContinuous(); X_source, _, _ = source.load()
target = next(n for n in X_source.columns if X_source.get_column(n).dtype.is_numeric() and X_source.get_column(n).n_unique() > 2)
m = MultidimSemiSyntheticDataset(source, n_treatments=3, confounding_loadings=[1.0,0.5,0.0], n_levels=[None,None,3], target_columns=[None,target,None], treatment_correlation=0.4, target_confounding_bias=0.5, random_state=0)
X, t, y = m.load(); print(t.head(), m.prognostic_fit_r2_, m.feature_fit_r2_, m.confounding_bias_ratio_)
"
```

Expected: prints a treatment head with `t_0`, `t_1` floats and `t_2` levels, R² values in `[0, 1]`, and a bias ratio near 0.5. If this takes longer than about a minute, note the timing in the commit message; do not change the metric constants.

- [ ] **Step 7: Full suite and lint**

Run: `cd ~/orca/workspaces/skcausal/manatee && PYTHONPATH=src python -m pytest tests --ignore=tests/datasets/test_all_datasets.py -q -p no:cacheprovider`
Expected: everything passes except the pre-existing collection errors listed in Global Constraints (density and "all objects" tests). Confirm those errors match the known `Float16`/skbase messages and nothing else.

Run: `cd ~/orca/workspaces/skcausal/manatee && ruff check src tests && ruff format --check src/skcausal/datasets/semi_synthetic.py src/skcausal/datasets/semi_synthetic_continuous.py src/skcausal/datasets/semi_synthetic_multidim.py src/skcausal/datasets/__init__.py tests/datasets/test_semi_synthetic_multidim_dgp.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_base.py tests/test_lookup.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/skcausal/datasets/__init__.py tests/test_lookup.py docs/research/semi_synthetic_datasets.qmd
git commit -m "feat(multidim): export MultidimSemiSyntheticDataset and document it"
```

---

## Self-Review Notes

- **Spec coverage.** Public API and per-component broadcasting: Task 3 and Task 4. Source split and validation: Task 1 (hook) and Task 4 (`_split_source`). Fitted confounder scores and baseline surface: Task 1 and Task 4. Assignment law, released transforms, joint log-probability, sampling: Task 4 (continuous) and Task 5 (categorical). Outcome surface with bounded cross terms: Task 4. Calibration with frozen common random numbers and enumerated discrete part: Task 4, exercised for categorical in Task 5. Grid: Task 4 and Task 5. Feature mimicking: Task 6. Files, exports, docs, discovery: Task 7. Extension boundaries need no code.
- **Type consistency.** `_log_density(latent_continuous, levels, means, cut_points)` is used with paired `(n, ·)` inputs in `_log_prob` and broadcast `(m, 1, ·)` vs `(1, n, k)` inputs in `_bias_at_strength`; `_gaussian_logpdf` and `_interval_log_mass` accept both. `cut_points` is always a `dict[int, ndarray]`. `_component_bases` takes `(n, k_C)` and `(n, k_D)` arrays in both call sites.
- **Temporary guards.** Task 4 rejects `n_levels` and `target_columns`; Task 5 narrows the guard to `target_columns`; Task 6 removes it. No guard survives the plan.
