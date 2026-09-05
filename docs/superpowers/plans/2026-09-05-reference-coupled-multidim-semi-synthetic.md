# Reference-Coupled Multidimensional Semi-Synthetic Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sampled multidimensional treatment model with a deterministic, seeded dataset that turns each regression source outcome into one treatment and couples whole source rows through an exchangeable Gaussian copula.

**Architecture:** Keep the existing multi-source loading hook in `BaseSemiSyntheticDataset`; put all source validation, copula coupling, mean-regressor scores, Dirichlet weighting, and additive spline effects in `MultidimSemiSyntheticDataset`. Construction performs the only treatment assembly, while `load()` is fixed and both public resampling and treatment-density APIs are disabled.

**Tech Stack:** Python 3.10+, NumPy, Polars, SciPy rank utilities, scikit-learn estimator cloning and `SplineTransformer`, pytest, Quarto.

**Spec:** `docs/superpowers/specs/2026-09-05-reference-coupled-multidim-semi-synthetic-design.md`

## Global Constraints

- Support continuous, finite, one-column regression outcomes only; categorical source outcomes and discretization remain source-dataset responsibilities for a later change.
- Do not sample or perturb treatment values: every `t_j` must be an unchanged selected source outcome.
- Preserve every selected source's complete covariate/outcome row while coupling sources.
- `treatment_correlation` is one exchangeable latent Gaussian-copula parameter, with `-1 / (P - 1) < rho < 1` for `P > 1` and `rho == 0` for `P == 1`.
- `confounding_concentration=None` means exact weights `1 / P`; a positive finite scalar means one seeded `Dirichlet(alpha * ones(P))` draw.
- Accept ordinary cloneable mean regressors and do not require `sample` or `log_prob` methods from them.
- Reuse installed NumPy, SciPy, Polars, and scikit-learn; add no dependency or source-configuration abstraction.
- Preserve scalar semi-synthetic behavior and the repository's pandas/Polars backend coercion.
- Remove obsolete multidimensional mixture-density, categorical-threshold, assignment-noise, bias-calibration, interaction, and heterogeneous-effect code instead of keeping two modes.

## File Structure

- Modify `src/skcausal/datasets/semi_synthetic_multidim.py`: the complete multidimensional implementation and its two small numerical helpers.
- Modify `src/skcausal/datasets/semi_synthetic.py`: retain only the shared multi-source loading lifecycle from the current draft; remove the abandoned source-column masking changes.
- Modify `tests/datasets/test_semi_synthetic_multidim_dgp.py`: replace old sampling/density/categorical tests with the reference-coupling contract.
- Modify `tests/datasets/test_semi_synthetic_base.py`: retain focused coverage of the shared multi-source hook.
- Modify `tests/datasets/test_semi_synthetic_helpers.py`: remove tests and imports for the abandoned column-mask helpers.
- Modify `tests/datasets/_semi_synthetic_test_utils.py`: retain the second unequal-size source fixture used by coupling tests.
- Modify `docs/research/semi_synthetic_datasets.qmd`: publish the mathematical construction, configuration, fixed lifecycle, and example.
- Delete `docs/superpowers/specs/2026-09-05-reference-coupled-multidim-semi-synthetic-design.md` and this plan after implementation, matching the repository policy that planning artifacts remain in Git history but are not rendered as site content.

---

### Task 1: Validated Gaussian-Copula Row Coupling

**Files:**
- Modify: `src/skcausal/datasets/semi_synthetic_multidim.py`
- Modify: `tests/datasets/test_semi_synthetic_multidim_dgp.py`

**Interfaces:**
- Consumes: `numpy.random.Generator`; one finite one-dimensional target array per source.
- Produces: `_exchangeable_correlation(n_components: int, correlation: float) -> np.ndarray` and `_coupled_orders(targets: list[np.ndarray], correlation: float, rng: np.random.Generator) -> tuple[list[np.ndarray], np.ndarray]`.

- [ ] **Step 1: Replace the obsolete numerical-helper tests with failing coupling tests**

```python
import numpy as np
import pytest
from scipy.stats import rankdata

from skcausal.datasets.semi_synthetic_multidim import (
    _coupled_orders,
    _exchangeable_correlation,
)


def test_exchangeable_correlation_validates_the_positive_definite_range():
    np.testing.assert_allclose(
        _exchangeable_correlation(3, -0.49),
        [[1.0, -0.49, -0.49], [-0.49, 1.0, -0.49], [-0.49, -0.49, 1.0]],
    )
    with pytest.raises(ValueError, match="-0.5 < treatment_correlation < 1"):
        _exchangeable_correlation(3, -0.5)
    with pytest.raises(ValueError, match="must be zero"):
        _exchangeable_correlation(1, 0.1)


@pytest.mark.parametrize("rho, direction", [(0.9, 1), (-0.9, -1)])
def test_coupled_orders_are_permutations_with_the_requested_rank_direction(
    rho, direction
):
    targets = [np.linspace(-2.0, 2.0, 2000), np.linspace(7.0, -1.0, 2000)]
    orders, latent = _coupled_orders(
        targets, rho, np.random.default_rng(9)
    )
    coupled = np.column_stack([target[order] for target, order in zip(targets, orders)])

    for order in orders:
        np.testing.assert_array_equal(np.sort(order), np.arange(2000))
    assert latent.shape == (2000, 2)
    achieved = np.corrcoef(rankdata(coupled, axis=0), rowvar=False)[0, 1]
    assert direction * achieved > 0.8


def test_coupled_orders_break_target_ties_reproducibly():
    targets = [np.repeat([0.0, 1.0], 10), np.arange(20.0)]
    first, _ = _coupled_orders(targets, 0.2, np.random.default_rng(5))
    second, _ = _coupled_orders(targets, 0.2, np.random.default_rng(5))
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
```

- [ ] **Step 2: Run the helper tests and verify the new coupling function is absent**

Run: `uv run pytest tests/datasets/test_semi_synthetic_multidim_dgp.py -q -k "exchangeable or coupled_orders"`

Expected: collection or test failure because `_coupled_orders` is not defined, and because the existing matrix helper does not validate the admissible range.

- [ ] **Step 3: Validate the existing matrix helper and add the coupling helper**

Keep the obsolete helpers until Task 2 replaces their only caller, so this intermediate commit leaves the existing class importable. Replace `_exchangeable_correlation` and add `_coupled_orders`:

```python
def _exchangeable_correlation(n_components, correlation):
    rho = float(correlation)
    if not np.isfinite(rho):
        raise ValueError("treatment_correlation must be finite.")
    if n_components == 1:
        if rho != 0.0:
            raise ValueError(
                "treatment_correlation must be zero with one source dataset."
            )
        return np.ones((1, 1))
    lower = -1.0 / (n_components - 1)
    if not lower < rho < 1.0:
        raise ValueError(
            "treatment_correlation must satisfy "
            f"{lower:g} < treatment_correlation < 1 for {n_components} sources."
        )
    matrix = np.full((n_components, n_components), rho)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _coupled_orders(targets, correlation, rng):
    arrays = [np.asarray(target, dtype=float).reshape(-1) for target in targets]
    if not arrays or len({array.size for array in arrays}) != 1:
        raise ValueError("Coupling requires equally sized source targets.")
    covariance = _exchangeable_correlation(len(arrays), correlation)
    latent = rng.multivariate_normal(
        np.zeros(len(arrays)), covariance, size=arrays[0].size
    ).reshape(arrays[0].size, len(arrays))
    orders = []
    for index, target in enumerate(arrays):
        latent_order = np.argsort(latent[:, index], kind="stable")
        source_order = np.lexsort((rng.random(target.size), target))
        order = np.empty(target.size, dtype=int)
        order[latent_order] = source_order
        orders.append(order)
    return orders, latent
```

- [ ] **Step 4: Run the focused helper tests**

Run: `uv run pytest tests/datasets/test_semi_synthetic_multidim_dgp.py -q -k "exchangeable or coupled_orders"`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the independently testable coupling primitive**

```bash
git add src/skcausal/datasets/semi_synthetic_multidim.py tests/datasets/test_semi_synthetic_multidim_dgp.py
git commit -m "feat: add Gaussian copula row coupling"
```

---

### Task 2: Fixed Multisource Dataset and Mean-Regressor Confounding

**Files:**
- Modify: `src/skcausal/datasets/semi_synthetic_multidim.py`
- Modify: `tests/datasets/test_semi_synthetic_multidim_dgp.py`
- Modify: `tests/datasets/_semi_synthetic_test_utils.py`

**Interfaces:**
- Consumes: the Task 1 helpers, the existing `BaseSemiSyntheticDataset._combine_sources(sources, rng)` hook, scikit-learn regressors with `fit(X, y)` and `predict(X)`, and source triples `(X, source_treatment, source_outcome)`.
- Produces: the approved `MultidimSemiSyntheticDataset` constructor, fitted attributes `source_row_indices_`, `source_scores_`, `confounding_weights_`, `source_mu0_`, `treatment_correlation_matrix_`, `achieved_pearson_correlation_`, and `achieved_spearman_correlation_`, plus fixed `load()`, `predict_y()`, `predict()`, and `get_grid()` behavior.

- [ ] **Step 1: Replace old DGP tests with failing public-contract tests**

Use the existing `ToyRealDataset` and `SecondToyRealDataset`; keep their categorical covariates to prove that preprocessing belongs to an ordinary sklearn pipeline:

```python
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _mean_regressor():
    return make_pipeline(
        make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include=object),
            ),
            remainder=StandardScaler(),
        ),
        LinearRegression(),
    )


def _dataset(**overrides):
    params = {
        "real_datasets": [ToyRealDataset(), SecondToyRealDataset()],
        "regressors": _mean_regressor(),
        "outcome_noise_scale": 0.0,
        "random_state": 13,
    }
    params.update(overrides)
    return MultidimSemiSyntheticDataset(**params)


def test_source_outcomes_become_fixed_treatments_without_breaking_rows():
    sources = [ToyRealDataset(), SecondToyRealDataset()]
    dataset = _dataset(real_datasets=sources)
    X, treatment, outcome = dataset.load()

    assert treatment.columns == ["t_0", "t_1"]
    assert X.columns == [
        "d0_age", "d0_income", "d0_region",
        "d1_height", "d1_score", "d1_group",
    ]
    for source_index, source in enumerate(sources):
        source_X, _, source_y = source.load()
        rows = dataset.source_row_indices_[source_index]
        expected_y = source_y.to_numpy().reshape(-1)[rows]
        np.testing.assert_allclose(treatment[f"t_{source_index}"], expected_y)
        for original in source_X.columns:
            np.testing.assert_array_equal(
                X[f"d{source_index}_{original}"].to_numpy(),
                source_X[original].to_numpy()[rows],
            )
    np.testing.assert_allclose(dataset.predict_y(X, treatment), outcome)
    assert all(left.equals(right) for left, right in zip(dataset.load(), dataset.load()))


def test_copula_diagnostics_and_default_confounding_baseline_are_exposed():
    dataset = _dataset(treatment_correlation=0.7)
    X, treatment, _ = dataset.load()

    np.testing.assert_allclose(dataset.confounding_weights_, [0.5, 0.5])
    raw = dataset.source_scores_ @ np.sqrt(dataset.confounding_weights_)
    expected = (raw - raw.mean()) / raw.std(ddof=0)
    np.testing.assert_allclose(dataset.source_mu0_, expected)
    np.testing.assert_allclose(dataset.treatment_correlation_matrix_, [[1, .7], [.7, 1]])
    np.testing.assert_allclose(
        dataset.achieved_pearson_correlation_,
        np.corrcoef(treatment.to_numpy(), rowvar=False),
    )
    assert dataset.achieved_spearman_correlation_.shape == (2, 2)
    np.testing.assert_allclose(dataset.predict_y(X, treatment).mean(), dataset.source_mu0_.mean(), atol=1e-12)


def test_dirichlet_weights_and_all_outputs_are_seeded_once_at_construction():
    first = _dataset(confounding_concentration=0.4, outcome_noise_scale=0.3)
    again = _dataset(confounding_concentration=0.4, outcome_noise_scale=0.3)
    other = _dataset(
        confounding_concentration=0.4, outcome_noise_scale=0.3, random_state=14
    )

    assert np.all(first.confounding_weights_ > 0.0)
    np.testing.assert_allclose(first.confounding_weights_.sum(), 1.0)
    np.testing.assert_allclose(first.confounding_weights_, again.confounding_weights_)
    assert all(left.equals(right) for left, right in zip(first.load(), again.load()))
    assert any(not left.equals(right) for left, right in zip(first.load(), other.load()))


def test_one_regressor_per_source_is_cloned_and_predict_is_available():
    configured = [_mean_regressor(), _mean_regressor()]
    dataset = _dataset(regressors=configured)
    X, _, _ = dataset.load()

    assert all(fitted is not source for fitted, source in zip(dataset.regressors_, configured))
    assert dataset.predict(X, dataset.get_grid(9).head(3)).shape == (3,)


def test_additive_spline_response_is_centered_and_supports_counterfactuals():
    dataset = _dataset(treatment_effect_scale=2.0)
    X, treatment, _ = dataset.load()
    observed = dataset._evaluate_treatment_effect(treatment)
    shifted = treatment.with_columns((pl.col("t_0") + 0.25).alias("t_0"))

    assert observed.mean() == pytest.approx(0.0, abs=1e-12)
    assert not np.allclose(dataset.predict_y(X, treatment), dataset.predict_y(X, shifted))
    grid = dataset.get_grid(100)
    assert grid.columns == ["t_0", "t_1"]
    assert grid.shape == (100, 2)
```

- [ ] **Step 2: Run the new public-contract tests and verify the constructor fails**

Run: `uv run pytest tests/datasets/test_semi_synthetic_multidim_dgp.py -q -k "source_outcomes or diagnostics or dirichlet or spline_response"`

Expected: failures because the current constructor does not accept `regressors` and still samples treatments.

- [ ] **Step 3: Replace the old class constructor and source assembly**

Replace the entire old class rather than leaving a compatibility mode. The constructor stores only the approved parameters and validates scalar controls before delegating to the base lifecycle:

```python
def __init__(
    self,
    real_datasets,
    regressors,
    treatment_correlation=0.0,
    confounding_concentration=None,
    n_spline_knots=6,
    spline_degree=3,
    treatment_effect_scale=1.0,
    outcome_noise_scale=1.0,
    random_state=42,
):
    if isinstance(real_datasets, BaseDataset):
        raise TypeError("real_datasets must be a non-empty sequence of datasets.")
    datasets = _validate_source_datasets(real_datasets, name="real_datasets")
    _exchangeable_correlation(len(datasets), treatment_correlation)
    if confounding_concentration is not None and (
        not np.isfinite(float(confounding_concentration))
        or float(confounding_concentration) <= 0.0
    ):
        raise ValueError("confounding_concentration must be positive and finite.")
    if not isinstance(n_spline_knots, (int, np.integer)) or n_spline_knots < 2:
        raise ValueError("n_spline_knots must be an integer at least 2.")
    if not isinstance(spline_degree, (int, np.integer)) or spline_degree < 0:
        raise ValueError("spline_degree must be a non-negative integer.")
    for name, value in {
        "treatment_effect_scale": treatment_effect_scale,
        "outcome_noise_scale": outcome_noise_scale,
    }.items():
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    self.real_datasets = real_datasets
    self.regressors = regressors
    self.treatment_correlation = treatment_correlation
    self.confounding_concentration = confounding_concentration
    self.n_spline_knots = n_spline_knots
    self.spline_degree = spline_degree
    self.treatment_effect_scale = treatment_effect_scale
    self.outcome_noise_scale = outcome_noise_scale
    super().__init__(real_dataset=datasets, random_state=random_state)
```

After replacing the class, delete `_per_component`, `_validate_loadings`, `_validate_levels`, `_interval_log_mass`, `_gaussian_logpdf`, `_conditional_gaussian`, all old assignment/density/calibration methods, and their now-unused imports. Keep `_source_prefix`, `_numeric_column`, `_exchangeable_correlation`, and `_coupled_orders`. The final module imports are:

```python
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.preprocessing import SplineTransformer

from skcausal.datasets.base import BaseDataset
from skcausal.datasets.semi_synthetic import (
    BaseSemiSyntheticDataset,
    _validate_source_datasets,
)
```

Implement `_combine_sources` so `source_row_indices_[j]` is the final output-row-to-original-source-row mapping:

```python
def _combine_sources(self, sources, rng):
    n_rows = min(X.height for X, _, _ in sources)
    selected_X, selected_y, selected_rows = [], [], []
    for index, (X, _, outcome) in enumerate(sources):
        outcome = self._coerce_backend_frame(outcome, backend="polars")
        if outcome.width != 1 or outcome.height != X.height:
            raise ValueError(
                f"Source dataset {index} must provide one outcome per covariate row."
            )
        y = _numeric_column(outcome, outcome.columns[0], f"outcome of source dataset {index}")
        rows = (
            np.arange(X.height)
            if X.height == n_rows
            else np.sort(rng.choice(X.height, n_rows, replace=False))
        )
        selected_X.append(X[rows])
        selected_y.append(y[rows])
        selected_rows.append(rows)
    orders, self.copula_latent_ = _coupled_orders(
        selected_y, self.treatment_correlation, rng
    )
    self.source_row_indices_ = [rows[order] for rows, order in zip(selected_rows, orders)]
    self.source_feature_names_ = [list(frame.columns) for frame in selected_X]
    self.source_columns_ = [
        [f"d{j}_{name}" for name in names]
        for j, names in enumerate(self.source_feature_names_)
    ]
    frames = [
        frame[order].rename(dict(zip(frame.columns, self.source_columns_[j])))
        for j, (frame, order) in enumerate(zip(selected_X, orders))
    ]
    coupled_y = np.column_stack(
        [values[order] for values, order in zip(selected_y, orders)]
    )
    self.treatment_columns_ = [f"t_{j}" for j in range(len(sources))]
    self.column_types = {name: "continuous" for name in self.treatment_columns_}
    self._fixed_treatments = pl.DataFrame(
        dict(zip(self.treatment_columns_, coupled_y.T))
    )
    self.treatment_correlation_matrix_ = _exchangeable_correlation(
        len(sources), self.treatment_correlation
    )
    self.achieved_pearson_correlation_ = _correlation_matrix(coupled_y)
    self.achieved_spearman_correlation_ = _correlation_matrix(
        rankdata(coupled_y, axis=0)
    )
    return pl.concat(frames, how="horizontal_extend")
```

Add the stable-shape diagnostic helper; Task 3 tightens `_numeric_column` at the validation boundary:

```python
def _correlation_matrix(values):
    values = np.asarray(values, dtype=float)
    if values.shape[1] == 1:
        return np.ones((1, 1))
    return np.corrcoef(values, rowvar=False)
```

- [ ] **Step 4: Implement regressor broadcasting, source score evaluation, and the weighted baseline**

Use `sklearn.base.clone`; pass source blocks as pandas frames with their original column names so a normal sklearn preprocessing pipeline works:

```python
def _regressor_list(self):
    if hasattr(self.regressors, "fit") and hasattr(self.regressors, "predict"):
        configured = [self.regressors] * len(self.source_columns_)
    else:
        try:
            configured = list(self.regressors)
        except TypeError as error:
            raise TypeError(
                "regressors must be one estimator or one estimator per source."
            ) from error
        if len(configured) != len(self.source_columns_):
            raise ValueError("regressors must contain one estimator per source.")
    try:
        return [clone(regressor) for regressor in configured]
    except TypeError as error:
        raise TypeError("Every regressor must be a cloneable sklearn estimator.") from error


def _source_frames(self, X):
    return [
        X.select(prefixed)
        .rename(dict(zip(prefixed, original)))
        .to_pandas()
        for prefixed, original in zip(self.source_columns_, self.source_feature_names_)
    ]


def _score_matrix(self, X):
    columns = []
    for regressor, frame, center, scale in zip(
        self.regressors_,
        self._source_frames(X),
        self.score_means_,
        self.score_scales_,
    ):
        prediction = np.asarray(regressor.predict(frame), dtype=float).reshape(-1)
        if prediction.size != X.height or not np.isfinite(prediction).all():
            raise ValueError("Each regressor must predict one finite value per row.")
        columns.append((prediction - center) / scale)
    return np.column_stack(columns)


def _prepare_dgp(self, X, rng):
    self.regressors_ = self._regressor_list()
    self.target_means_ = self._fixed_treatments.to_numpy().mean(axis=0)
    self.target_scales_ = self._fixed_treatments.to_numpy().std(axis=0, ddof=0)
    frames = self._source_frames(X)
    fitted = []
    for index, (regressor, frame) in enumerate(zip(self.regressors_, frames)):
        normalized_y = (
            self._fixed_treatments
            .get_column(self.treatment_columns_[index])
            .to_numpy()
            - self.target_means_[index]
        ) / self.target_scales_[index]
        if "random_state" in regressor.get_params(deep=False) and regressor.get_params(deep=False)["random_state"] is None:
            regressor.set_params(random_state=int(rng.integers(np.iinfo(np.int32).max)))
        regressor.fit(frame, normalized_y)
        prediction = np.asarray(regressor.predict(frame), dtype=float).reshape(-1)
        if prediction.size != self.n or not np.isfinite(prediction).all() or prediction.std(ddof=0) <= 0.0:
            raise ValueError("Each fitted regressor must produce nonconstant finite predictions.")
        fitted.append(prediction)
    fitted = np.column_stack(fitted)
    self.score_means_ = fitted.mean(axis=0)
    self.score_scales_ = fitted.std(axis=0, ddof=0)
    self.source_scores_ = (fitted - self.score_means_) / self.score_scales_
    if self.confounding_concentration is None:
        self.confounding_weights_ = np.full(len(self.regressors_), 1.0 / len(self.regressors_))
    else:
        self.confounding_weights_ = rng.dirichlet(
            np.full(len(self.regressors_), float(self.confounding_concentration))
        )
    raw = self.source_scores_ @ np.sqrt(self.confounding_weights_)
    self.baseline_mean_ = float(raw.mean())
    self.baseline_scale_ = float(raw.std(ddof=0))
    if not np.isfinite(self.baseline_scale_) or self.baseline_scale_ <= 0.0:
        raise ValueError("The weighted confounding baseline must be nonconstant.")
    self.source_mu0_ = (raw - self.baseline_mean_) / self.baseline_scale_
    self._fit_treatment_effect(rng)


def _baseline(self, X):
    raw = self._score_matrix(X) @ np.sqrt(self.confounding_weights_)
    return (raw - self.baseline_mean_) / self.baseline_scale_
```

- [ ] **Step 5: Implement the additive spline effect, fixed hooks, and grid**

```python
def _fit_treatment_effect(self, rng):
    observed = self._fixed_treatments.to_numpy()
    self.spline_transformers_ = []
    self.spline_coefficients_ = []
    raw_effect = np.zeros(self.n)
    for values in observed.T:
        unique_count = np.unique(np.round(values, 12)).size
        transformer = SplineTransformer(
            n_knots=min(self.n_spline_knots, max(2, unique_count)),
            degree=self.spline_degree,
            include_bias=False,
            extrapolation="continue",
        )
        basis = transformer.fit_transform(values.reshape(-1, 1))
        coefficients = rng.normal(
            scale=1.0 / np.sqrt(len(self.treatment_columns_) * basis.shape[1]),
            size=basis.shape[1],
        )
        self.spline_transformers_.append(transformer)
        self.spline_coefficients_.append(coefficients)
        raw_effect += basis @ coefficients
    self.treatment_effect_offset_ = float(raw_effect.mean())
    self.treatment_ranges_ = np.column_stack((observed.min(axis=0), observed.max(axis=0)))


def _evaluate_treatment_effect(self, treatment):
    values = treatment.select(self.treatment_columns_).to_numpy()
    raw = sum(
        transformer.transform(values[:, [index]]) @ coefficients
        for index, (transformer, coefficients) in enumerate(
            zip(self.spline_transformers_, self.spline_coefficients_)
        )
    )
    return self.treatment_effect_scale * (raw - self.treatment_effect_offset_)


def _sample_treatment(self, X, rng):
    return self._fixed_treatments


def _predict_y(self, X, treatment):
    return self._baseline(X) + self._evaluate_treatment_effect(treatment)


def _sample_outcome(self, mean, X, treatment, rng):
    return np.asarray(mean).reshape(-1) + rng.normal(
        scale=self.outcome_noise_scale, size=X.height
    )


def _get_grid(self, n):
    points = max(2, round(n ** (1.0 / len(self.treatment_columns_))))
    grid = None
    for name, (lower, upper) in zip(self.treatment_columns_, self.treatment_ranges_):
        component = pl.DataFrame({name: np.linspace(lower, upper, points)})
        grid = component if grid is None else grid.join(component, how="cross")
    return grid
```

Keep `_log_prob` as a private raising hook to satisfy the abstract base class. Public disabling is tested in Task 3. Add this global-contract fixture so cloning and changed seeds are exercised:

```python
@classmethod
def get_test_params(cls, parameter_set="default"):
    from sklearn.linear_model import LinearRegression

    from skcausal.datasets.kang_schafer import KangSchaferContinuous

    return [
        {
            "real_datasets": [
                KangSchaferContinuous(n=64, random_state=4),
                KangSchaferContinuous(n=80, random_state=5),
            ],
            "regressors": LinearRegression(),
            "treatment_correlation": 0.4,
            "random_state": 7,
        }
    ]
```

- [ ] **Step 6: Run the multidimensional tests and the global dataset contract**

Run: `uv run pytest tests/datasets/test_semi_synthetic_multidim_dgp.py tests/datasets/test_all_datasets.py -q`

Expected: all tests pass without a Polars horizontal-concatenation warning.

- [ ] **Step 7: Commit the fixed dataset implementation**

```bash
git add src/skcausal/datasets/semi_synthetic_multidim.py tests/datasets/test_semi_synthetic_multidim_dgp.py tests/datasets/_semi_synthetic_test_utils.py
git commit -m "feat: build fixed treatments from regression sources"
```

---

### Task 3: Validation, Disabled APIs, and Shared-Base Cleanup

**Files:**
- Modify: `src/skcausal/datasets/semi_synthetic.py`
- Modify: `src/skcausal/datasets/semi_synthetic_multidim.py`
- Modify: `tests/datasets/test_semi_synthetic_base.py`
- Modify: `tests/datasets/test_semi_synthetic_helpers.py`
- Modify: `tests/datasets/test_semi_synthetic_multidim_dgp.py`

**Interfaces:**
- Consumes: the Task 2 public constructor and fitted dataset.
- Produces: clear trust-boundary errors, public `sample(random_state)` and `log_prob(X, treatment)` failures, and a shared base that supports either one source or subclass-controlled multi-source combination without changing scalar score machinery.

- [ ] **Step 1: Add failing validation and disabled-API tests**

```python
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor


class BadRegressor(BaseEstimator):
    def __init__(self, kind="nan"):
        self.kind = kind

    def fit(self, X, y):
        return self

    def predict(self, X):
        if self.kind == "nan":
            return np.full(len(X), np.nan)
        return np.zeros((len(X), 2))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"real_datasets": []}, "non-empty sequence"),
        ({"real_datasets": ToyRealDataset()}, "non-empty sequence"),
        ({"regressors": []}, "one estimator per source"),
        ({"regressors": [object(), object()]}, "cloneable sklearn estimator"),
        ({"treatment_correlation": 1.0}, "treatment_correlation"),
        ({"treatment_correlation": -1.0}, "treatment_correlation"),
        ({"confounding_concentration": 0.0}, "confounding_concentration"),
        ({"confounding_concentration": np.inf}, "confounding_concentration"),
        ({"n_spline_knots": 1}, "n_spline_knots"),
        ({"spline_degree": -1}, "spline_degree"),
        ({"treatment_effect_scale": -1.0}, "treatment_effect_scale"),
        ({"outcome_noise_scale": -1.0}, "outcome_noise_scale"),
    ],
)
def test_multidim_rejects_invalid_configuration(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _dataset(**overrides)


def test_multidim_rejects_constant_mean_predictions():
    with pytest.raises(ValueError, match="nonconstant finite predictions"):
        _dataset(regressors=DummyRegressor(strategy="mean"))


@pytest.mark.parametrize("kind", ["nan", "two_columns"])
def test_multidim_rejects_non_scalar_or_nonfinite_predictions(kind):
    with pytest.raises(ValueError, match="finite predictions"):
        _dataset(regressors=BadRegressor(kind))


def test_multidim_disables_resampling_and_treatment_density():
    dataset = _dataset()
    X, treatment, _ = dataset.load()
    with pytest.raises(NotImplementedError, match="fixed at construction"):
        dataset.sample(4)
    with pytest.raises(NotImplementedError, match="does not define"):
        dataset.log_prob(X, treatment)


class InvalidOutcomeSource(BaseDataset):
    def __init__(self, outcome):
        self.outcome = outcome
        super().__init__()

    def _load(self):
        return (
            pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
            pl.DataFrame({"unused_t": [0.0, 0.0, 0.0, 0.0]}),
            self.outcome,
        )


@pytest.mark.parametrize(
    "outcome, message",
    [
        (pl.DataFrame({"a": [0., 1., 2., 3.], "b": [3., 2., 1., 0.]}), "exactly one outcome"),
        (pl.DataFrame({"y": ["a", "b", "c", "d"]}), "must be numeric"),
        (pl.DataFrame({"y": [0., 1., np.nan, 3.]}), "must be finite"),
        (pl.DataFrame({"y": [1., 1., 1., 1.]}), "at least two distinct"),
        (pl.DataFrame({"y": [0., 1., 2.]}), "one outcome per covariate row"),
    ],
)
def test_multidim_rejects_invalid_source_outcomes(outcome, message):
    with pytest.raises(ValueError, match=message):
        _dataset(real_datasets=[InvalidOutcomeSource(outcome)])


def test_single_source_requires_zero_correlation_and_has_matrix_diagnostics():
    single = _dataset(real_datasets=[ToyRealDataset()])
    assert single.achieved_pearson_correlation_.shape == (1, 1)
    assert single.achieved_spearman_correlation_.shape == (1, 1)
    with pytest.raises(ValueError, match="must be zero"):
        _dataset(real_datasets=[ToyRealDataset()], treatment_correlation=0.1)
```

- [ ] **Step 2: Run focused tests and verify missing cases fail**

Run: `uv run pytest tests/datasets/test_semi_synthetic_multidim_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py -q`

Expected: failures for public `sample`, public `log_prob`, and any target/regressor validation not yet covered by Task 2.

- [ ] **Step 3: Disable the two public APIs on the multidimensional class**

```python
def sample(self, random_state):
    raise NotImplementedError(
        "MultidimSemiSyntheticDataset is fixed at construction; create a new "
        "instance with another random_state instead."
    )


def log_prob(self, X, treatment):
    raise NotImplementedError(
        "Mean regressors do not define p(T | X), so this dataset does not define log_prob."
    )


def _log_prob(self, X, treatment):
    raise NotImplementedError
```

Make `_numeric_column` reject non-finite data and constant data before row selection or fitting:

```python
def _numeric_column(frame, name, label):
    series = frame.get_column(name)
    if not series.dtype.is_numeric():
        raise ValueError(f"The {label} must be numeric.")
    values = series.cast(pl.Float64).to_numpy().astype(float)
    if not np.isfinite(values).all():
        raise ValueError(f"The {label} must be finite.")
    if np.unique(values).size < 2:
        raise ValueError(f"The {label} must have at least two distinct values.")
    return values
```

Keep estimator clone, prediction shape, prediction finiteness, and nonconstant-baseline checks at construction so failures occur before a partial dataset can be used.

- [ ] **Step 4: Reduce the shared base diff to the multi-source lifecycle only**

In `semi_synthetic.py`, keep `_validate_source_datasets`, `real_datasets_`, the three derived random streams, and `_combine_sources`. Restore the scalar score helpers to their pre-draft behavior by applying this inverse of the abandoned masking edits; delete the complete `_encoded_column_origins` and `_library_column_masks` definitions between the shown hunks:

```diff
-def _ridge_projections(library, targets, column_masks=None):
+def _ridge_projections(library, targets):
@@
-    if column_masks is None:
-        column_masks = [np.ones(library.shape[1], dtype=bool)] * targets.shape[1]
-    if len(column_masks) != targets.shape[1]:
-        raise ValueError("column_masks must provide one mask per target column.")
@@
-        mask = np.asarray(column_masks[index], dtype=bool)
-        if mask.shape != (library.shape[1],):
-            raise ValueError("Each column mask must cover every library column.")
-        active = (scale > 0.0) & mask
+        active = scale > 0.0
@@
-def _fit_causal_score_map(
-    X, rng, fitted_confounders=None, fitted_confounder_columns=None
-):
+def _fit_causal_score_map(X, rng, fitted_confounders=None):
@@
-    column_masks = None
-    if fitted_confounder_columns is not None:
-        if fitted_confounders is None:
-            raise ValueError("fitted_confounder_columns requires fitted_confounders.")
-        for group in fitted_confounder_columns:
-            unknown = set(group) - set(columns)
-            if unknown:
-                raise ValueError(
-                    f"fitted_confounder_columns names unknown covariates {unknown}."
-                )
-        column_masks = _library_column_masks(
-            _encoded_column_origins(encoder, numeric_columns, categorical_columns),
-            pair_indices,
-            fitted_confounder_columns,
-        )
@@
-            projection, fitted_r2 = _ridge_projections(
-                library, fitted_confounders, column_masks=column_masks
-            )
+            projection, fitted_r2 = _ridge_projections(library, fitted_confounders)
@@
-        fitted_confounder_columns=None,
@@
-            fitted_confounder_columns=fitted_confounder_columns,
```

Delete `_encoded_column_origins`, `_library_column_masks`, the `column_masks` branches, and `fitted_confounder_columns`. Delete their imports and three tests from `test_semi_synthetic_helpers.py`; those abstractions belonged to the discarded score-map approach and have no remaining caller. Retain the multi-source tests in `test_semi_synthetic_base.py` and change their concatenation to `pl.concat(frames, how="horizontal_extend")`, matching the multidimensional implementation and avoiding the installed Polars deprecation warning.

- [ ] **Step 5: Run scalar and multidimensional regression tests together**

Run: `uv run pytest tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_continuous_dgp.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_multidim_dgp.py -q`

Expected: all selected tests pass with no new warnings.

- [ ] **Step 6: Run formatting/lint checks on changed Python files**

Run: `uv run ruff check src/skcausal/datasets/semi_synthetic.py src/skcausal/datasets/semi_synthetic_multidim.py tests/datasets/_semi_synthetic_test_utils.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_multidim_dgp.py`

Expected: exit code 0.

- [ ] **Step 7: Commit validation and cleanup**

```bash
git add src/skcausal/datasets/semi_synthetic.py src/skcausal/datasets/semi_synthetic_multidim.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_multidim_dgp.py
git commit -m "test: enforce fixed multidim dataset contract"
```

---

### Task 4: Mathematical Documentation and Final Verification

**Files:**
- Modify: `docs/research/semi_synthetic_datasets.qmd`
- Delete: `docs/superpowers/specs/2026-09-05-reference-coupled-multidim-semi-synthetic-design.md`
- Delete: `docs/superpowers/plans/2026-09-05-reference-coupled-multidim-semi-synthetic.md`

**Interfaces:**
- Consumes: the final Task 3 constructor, fitted attributes, and equations from the approved spec.
- Produces: rendered public documentation matching the implementation; no planning artifacts in the published documentation tree.

- [ ] **Step 1: Replace the old multidimensional narrative and executable example**

The example must use two regression sources and an ordinary point regressor:

```python
from sklearn.linear_model import LinearRegression

from skcausal.datasets import KangSchaferContinuous, MultidimSemiSyntheticDataset

multidim = MultidimSemiSyntheticDataset(
    real_datasets=[
        KangSchaferContinuous(n=500, random_state=1),
        KangSchaferContinuous(n=700, random_state=2),
    ],
    regressors=LinearRegression(),
    treatment_correlation=0.6,
    confounding_concentration=None,
    random_state=3,
)
X, treatment, outcome = multidim.load()
```

State explicitly that source treatment columns are ignored, each source outcome is retained in original units as `t_j`, and `sample`/`log_prob` are unavailable.

- [ ] **Step 2: Document the complete mathematical construction**

Include these equations and their implementation meanings:

```markdown
Let $n=\min_j n_j$ and let $\widetilde Y_{ij}=(Y_{ij}-\bar Y_j)/s_{Y_j}$.
Fit $\widehat m_j$ to $(X_j,\widetilde Y_j)$.

For $R_\rho=(1-\rho)I+\rho\mathbf 1\mathbf 1^\top$, draw
$G_i\sim\mathcal N(0,R_\rho)$ and rank-match whole source rows sorted by $Y_j$
to the ranks of $G_{ij}$. This preserves every source marginal and every
within-source $(X_j,Y_j)$ pair. The configured $\rho$ is latent; the object
therefore records achieved Pearson and Spearman matrices as diagnostics.

After coupling, standardize the fitted scores as
$s_{ij}=(\widehat m_j(X_{ij})-\overline{\widehat m_j})/
\operatorname{sd}(\widehat m_j)$. Use $w_j=1/P$ when concentration is omitted,
or draw $w\sim\operatorname{Dirichlet}(\alpha\mathbf 1_P)$ once. Then
$\mu_0(X_i)=\operatorname{standardize}(\sum_j\sqrt{w_j}s_{ij})$.
Large $\alpha$ concentrates weights near equal importance; small $\alpha$
allows a few treatment scores to dominate. With correlated scores, the weights
are loadings rather than a unique variance decomposition.

With one fitted B-spline basis per component and
$\beta_{jk}\sim\mathcal N(0,1/(PK_j))$, define
$g(t)=\lambda[\sum_j B_j(t_j)^\top\beta_j-
n^{-1}\sum_i\sum_j B_j(T_{ij})^\top\beta_j]$ and
$Y_i=\mu_0(X_i)+g(T_i)+\sigma_Y\varepsilon_i$.
```

Explain why a 45-degree value rotation does not control observed correlation in general and would change treatment marginals, whereas rank coupling leaves the observed source outcomes unchanged. Remove descriptions of `n_levels`, mixture densities, assignment noise, calibration, interactions, and heterogeneous effects from the multidimensional section and configuration table.

Use a compact configuration table containing exactly `real_datasets`, `regressors`, `treatment_correlation`, `confounding_concentration`, `n_spline_knots`, `spline_degree`, `treatment_effect_scale`, `outcome_noise_scale`, and `random_state`. State that categorical treatments require a future categorical-aware implementation; a discretized regression target is not silently treated as categorical here.

- [ ] **Step 3: Render the research page**

Run: `quarto render docs/research/semi_synthetic_datasets.qmd`

Expected: exit code 0 and no broken Python example or cross-reference.

- [ ] **Step 4: Run the complete test suite and inspect the final diff**

Run: `uv run pytest -q`

Expected: full suite passes.

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: only the intended documentation change and the two planning-artifact deletions remain unstaged.

- [ ] **Step 5: Remove planning artifacts and commit the public documentation**

Use `apply_patch` to delete the approved spec and this plan, then:

```bash
git add docs/research/semi_synthetic_datasets.qmd docs/superpowers/specs/2026-09-05-reference-coupled-multidim-semi-synthetic-design.md docs/superpowers/plans/2026-09-05-reference-coupled-multidim-semi-synthetic.md
git commit -m "docs: explain reference-coupled treatment generation"
```

- [ ] **Step 6: Verify the committed state**

Run: `git status --short`

Expected: empty output.

Run: `git log -4 --oneline`

Expected: the coupling, fixed-dataset, validation, and documentation commits are present, with the design specification still recoverable from Git history.
