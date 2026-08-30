# Semi-Synthetic Observational DGP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add clone-safe semi-synthetic dataset classes that preserve covariates from a supplied `BaseDataset`, generate categorical or scalar-continuous observational treatments and Gaussian outcomes, and expose exact assignment and outcome oracles.

**Architecture:** `BaseSemiSyntheticDataset` owns source-dataset loading, frozen structural preparation, stochastic replication, backend coercion, and the public `log_prob` contract. `CategoricalSemiSyntheticDataset` and `ContinuousSemiSyntheticDataset` each implement one complete coupled DGP behind protected hooks; shared score construction, numerical density, outcome, and calibration operations remain private functions in `semi_synthetic.py`.

**Tech Stack:** Python >=3.10,<3.14, NumPy, pandas, Polars, scikit-learn preprocessing, scikit-base, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-30-semi-synthetic-dgp-design.md`

## Global Constraints

- `real_dataset` must be a cloneable `BaseDataset`; built-in DGPs use only the first frame returned by `real_dataset.load()`.
- Released covariates retain source values, columns, and row order; all encoding and imputation are internal.
- The public oracle API is `predict_y(X, treatment)` for `m(a, x)` and `log_prob(X, treatment)` for exact conditional log-mass/log-density.
- `sample(random_state)` changes treatment and outcome draws without changing frozen structural state or mutating `load()` outputs.
- Initial built-ins cover categorical and scalar-continuous treatments with Gaussian continuous outcomes.
- Categorical outputs use repository-compatible string levels from `"0"` through `str(K - 1)` stored with categorical dtype.
- Confounding and overlap remain separate: `confounding_strength` or normalized `target_confounding_bias` controls shared prognostic selection; `randomized_weight` and continuous treatment noise control overlap.
- Do not add SciPy or another dependency; use NumPy and Python's `statistics.NormalDist` for numerical work.
- Existing `SemiSyntheticRegressor` and `SemiSyntheticClassifier` behavior must remain unchanged.

## File Map

- Create `src/skcausal/datasets/semi_synthetic.py`: base lifecycle plus private score, outcome, density, marginal-calibration, and strength-calibration helpers.
- Create `src/skcausal/datasets/semi_synthetic_categorical.py`: complete categorical assignment and outcome DGP.
- Create `src/skcausal/datasets/semi_synthetic_continuous.py`: complete scalar-continuous assignment and outcome DGP.
- Modify `src/skcausal/datasets/__init__.py`: export the three new public classes.
- Create `tests/datasets/__init__.py`: make the dataset-test directory importable for shared fixtures.
- Create `tests/datasets/_semi_synthetic_test_utils.py`: deterministic real-dataset fixture shared by the new tests.
- Create `tests/datasets/test_semi_synthetic_base.py`: base lifecycle, oracle wrapper, cloning, schema, and replication tests.
- Create `tests/datasets/test_semi_synthetic_helpers.py`: internal encoding, score, numerical probability, and calibration tests.
- Create `tests/datasets/test_semi_synthetic_categorical_dgp.py`: categorical DGP and calibration tests.
- Create `tests/datasets/test_semi_synthetic_continuous_dgp.py`: continuous DGP, transformation, density, and calibration tests.
- Modify `tests/test_lookup.py`: verify public dataset discovery.

---

### Task 1: Base semi-synthetic lifecycle and oracle contract

**Files:**
- Create: `src/skcausal/datasets/semi_synthetic.py`
- Create: `tests/datasets/__init__.py`
- Create: `tests/datasets/_semi_synthetic_test_utils.py`
- Create: `tests/datasets/test_semi_synthetic_base.py`

**Interfaces:**
- Consumes: `BaseDataset.load()`, `BaseDataset._coerce_backend_frame`, `BaseSyntheticDataset.predict_y`, and `BaseSyntheticDataset._to_polars`.
- Produces: abstract `BaseSemiSyntheticDataset(real_dataset, random_state=42)`, public `log_prob(X, treatment) -> np.ndarray`, public `sample(random_state) -> tuple[DataFrame, DataFrame, DataFrame]`, public `get_grid(n=100) -> DataFrame`, and protected hooks `_prepare_dgp`, `_sample_treatment`, `_log_prob`, `_predict_y`, `_sample_outcome`, `_get_grid`.

- [ ] **Step 1: Make dataset tests importable and add the deterministic source-dataset fixture**

Create an empty `tests/datasets/__init__.py`, then add:

```python
# tests/datasets/_semi_synthetic_test_utils.py
import numpy as np
import polars as pl

from skcausal.datasets.base import BaseDataset


class ToyRealDataset(BaseDataset):
    def __init__(self, include_missing=False):
        self.include_missing = include_missing
        super().__init__()

    def _load(self):
        age = [21.0, 25.0, 29.0, 33.0, 37.0, 41.0, 45.0, 49.0]
        if self.include_missing:
            age[2] = None
        X = pl.DataFrame(
            {
                "age": age,
                "income": [32.0, 38.0, 44.0, 50.0, 56.0, 62.0, 68.0, 74.0],
                "region": ["n", "s", "n", "e", "s", "e", "n", "s"],
            }
        )
        source_t = pl.DataFrame({"source_t": np.arange(8, dtype=float)})
        source_y = pl.DataFrame({"source_y": np.linspace(-1.0, 1.0, 8)})
        return X, source_t, source_y
```

- [ ] **Step 2: Write failing lifecycle and replication tests using a minimal test DGP**

```python
# tests/datasets/test_semi_synthetic_base.py
import numpy as np
import polars as pl
import pytest

from skcausal.datasets.semi_synthetic import BaseSemiSyntheticDataset
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


class ToySemiSyntheticDataset(BaseSemiSyntheticDataset):
    column_types = {"t": "continuous"}

    def _prepare_dgp(self, X, rng):
        self.offset_ = float(X.get_column("age").mean())

    def _sample_treatment(self, X, rng):
        values = X.get_column("income").to_numpy() / 50.0
        return pl.DataFrame({"t": values + rng.normal(scale=0.2, size=len(values))})

    def _log_prob(self, X, treatment):
        mean = X.get_column("income").to_numpy() / 50.0
        values = treatment.get_column("t").to_numpy()
        return -0.5 * ((values - mean) / 0.2) ** 2 - np.log(0.2 * np.sqrt(2 * np.pi))

    def _predict_y(self, X, treatment):
        age = X.get_column("age").to_numpy()
        values = treatment.get_column("t").to_numpy()
        return ((age - self.offset_) / 10.0 + values).reshape(-1, 1)

    def _sample_outcome(self, mean, X, treatment, rng):
        return np.asarray(mean) + rng.normal(scale=0.1, size=np.asarray(mean).shape)

    def _get_grid(self, n):
        return pl.DataFrame({"t": np.linspace(-1.0, 1.0, n)})


def test_base_preserves_source_covariates_and_oracle_shape():
    source = ToyRealDataset()
    dataset = ToySemiSyntheticDataset(source, random_state=7)
    X, treatment, outcome = dataset.load()

    assert X.equals(source.load()[0])
    assert dataset.real_dataset_ is not source
    assert treatment.columns == ["t"]
    assert outcome.columns == ["y"]
    assert dataset.log_prob(X, treatment).shape == (X.height, 1)
    assert dataset.predict_y(X, treatment).shape == (X.height, 1)
    assert dataset.get_grid(5).shape == (5, 1)


def test_sample_is_reproducible_and_does_not_mutate_load():
    dataset = ToySemiSyntheticDataset(ToyRealDataset(), random_state=11)
    loaded_before = dataset.load()
    first = dataset.sample(random_state=19)
    second = dataset.sample(random_state=19)
    loaded_after = dataset.load()

    for first_frame, second_frame in zip(first, second):
        assert first_frame.equals(second_frame)
    for before, after in zip(loaded_before, loaded_after):
        assert before.equals(after)
    assert not first[1].equals(loaded_before[1])


def test_base_rejects_prepare_sample_size_and_misaligned_oracle_rows():
    dataset = ToySemiSyntheticDataset(ToyRealDataset(), random_state=3)
    X, treatment, _ = dataset.load()

    with pytest.raises(ValueError, match="sample size"):
        dataset.prepare(n=10)
    with pytest.raises(ValueError, match="same number of rows"):
        dataset.log_prob(X.head(2), treatment.head(1))


def test_base_requires_a_dataset_object():
    with pytest.raises(TypeError, match="BaseDataset"):
        ToySemiSyntheticDataset(object(), random_state=3)
```

- [ ] **Step 3: Run the base tests and verify the missing module failure**

Run: `uv run pytest tests/datasets/test_semi_synthetic_base.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'skcausal.datasets.semi_synthetic'`.

- [ ] **Step 4: Implement the abstract lifecycle and public wrappers**

```python
# src/skcausal/datasets/semi_synthetic.py
from abc import abstractmethod

import numpy as np
import polars as pl

from skcausal.datasets.base import BaseDataset, BaseSyntheticDataset

__all__ = ["BaseSemiSyntheticDataset"]


class BaseSemiSyntheticDataset(BaseSyntheticDataset):
    outcome_columns = ("y",)

    def __init__(self, real_dataset: BaseDataset, random_state: int = 42):
        if not isinstance(real_dataset, BaseDataset):
            raise TypeError("real_dataset must be an instance of BaseDataset.")
        self.real_dataset = real_dataset
        super().__init__(n=0, random_state=random_state)
        self._prepare()

    def _prepare(self, n=None):
        if n is not None:
            raise ValueError(
                "BaseSemiSyntheticDataset derives its sample size from real_dataset "
                "and does not accept a sample size."
            )
        self.real_dataset_ = self.real_dataset.clone()
        X, _, _ = self.real_dataset_.load()
        X = self._coerce_backend_frame(X, backend="polars")
        if X.height == 0 or len(set(X.columns)) != X.width:
            raise ValueError("real_dataset must provide nonempty covariates with unique columns.")
        self.n = X.height
        structural_seed, sample_seed = np.random.SeedSequence(self.random_state).spawn(2)
        self._prepare_dgp(X, np.random.default_rng(structural_seed))
        treatment, outcome = self._draw_sample(X, sample_seed)
        self._covariates = X
        self._treatments = treatment
        self._outcomes = outcome
        return self

    def _draw_sample(self, X, seed):
        treatment_seed, outcome_seed = seed.spawn(2)
        treatment = self._to_polars(
            self._sample_treatment(X, np.random.default_rng(treatment_seed))
        )
        mean = self.predict_y(X, treatment)
        outcome = self._coerce_outcome_frame(
            self._sample_outcome(
                mean, X, treatment, np.random.default_rng(outcome_seed)
            )
        )
        return treatment, outcome

    def _coerce_outcome_frame(self, values):
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.shape != (self.n, len(self.outcome_columns)):
            raise ValueError("Outcome samples must match source rows and outcome columns.")
        return pl.DataFrame(
            {name: array[:, index] for index, name in enumerate(self.outcome_columns)}
        )

    def sample(self, random_state):
        treatment, outcome = self._draw_sample(
            self._covariates, np.random.SeedSequence(random_state)
        )
        return (
            self._coerce_backend_frame(self._covariates),
            self._coerce_backend_frame(
                treatment, column_types=self._get_treatment_column_types()
            ),
            self._coerce_backend_frame(outcome),
        )

    def log_prob(self, X, treatment):
        X = self._check_and_transform_X(X)
        treatment = self._check_and_transform_t(treatment)
        if self._get_n_samples(X) != self._get_n_samples(treatment):
            raise ValueError("log_prob requires X and treatment to have the same number of rows.")
        values = np.asarray(self._log_prob(X, treatment), dtype=float).reshape(-1, 1)
        if values.shape[0] != self._get_n_samples(X) or np.isnan(values).any():
            raise ValueError("_log_prob must return one non-NaN value per row.")
        return values

    def get_grid(self, n=100):
        return self._coerce_treatment_frame(self._get_grid(n))

    @abstractmethod
    def _prepare_dgp(self, X, rng):
        raise NotImplementedError

    @abstractmethod
    def _sample_treatment(self, X, rng):
        raise NotImplementedError

    @abstractmethod
    def _log_prob(self, X, treatment):
        raise NotImplementedError

    @abstractmethod
    def _sample_outcome(self, mean, X, treatment, rng):
        raise NotImplementedError

    @abstractmethod
    def _get_grid(self, n):
        raise NotImplementedError
```

Ensure `_check_and_transform_X` and `_check_and_transform_t` produce Polars frames for dataframe inputs and normalize NumPy inputs before invoking hooks.

- [ ] **Step 5: Run the base tests and the existing base-dataset tests**

Run: `uv run pytest tests/datasets/test_semi_synthetic_base.py tests/datasets/test_all_datasets.py -q`

Expected: all selected tests pass; abstract `BaseSemiSyntheticDataset` is not instantiated by discovery.

- [ ] **Step 6: Commit the base lifecycle**

```bash
git add src/skcausal/datasets/semi_synthetic.py tests/datasets/__init__.py tests/datasets/_semi_synthetic_test_utils.py tests/datasets/test_semi_synthetic_base.py
git commit -m "feat: add semi-synthetic dataset base"
```

### Task 2: Frozen nonlinear causal-role scores and Gaussian baseline helpers

**Files:**
- Modify: `src/skcausal/datasets/semi_synthetic.py`
- Create: `tests/datasets/test_semi_synthetic_helpers.py`

**Interfaces:**
- Consumes: Polars, pandas, or NumPy covariates plus a `np.random.Generator`.
- Produces: `_CausalScores`, `_CausalScoreMap.transform(X)`, `_fit_causal_score_map(X, rng) -> tuple[_CausalScoreMap, _CausalScores]`, `_BaselineSurface.evaluate(scores) -> tuple[mu0, confounding_signal]`, and `_fit_baseline_surface(scores, rng) -> _BaselineSurface`.

- [ ] **Step 1: Write failing score-map tests for mixed, missing, and new-category data**

```python
# tests/datasets/test_semi_synthetic_helpers.py
import numpy as np
import polars as pl
import pytest

from skcausal.datasets.semi_synthetic import (
    _fit_baseline_surface,
    _fit_causal_score_map,
)
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


def test_score_map_is_frozen_finite_and_handles_unknown_categories():
    X = ToyRealDataset(include_missing=True).load()[0]
    score_map, scores = _fit_causal_score_map(
        X, np.random.default_rng(4)
    )
    repeated = score_map.transform(X)
    new_X = X.with_columns(pl.lit("west").alias("region"))
    unseen = score_map.transform(new_X)

    for name in ("confounders", "prognostic", "treatment_only", "modifiers"):
        np.testing.assert_allclose(getattr(scores, name), getattr(repeated, name))
        assert np.isfinite(getattr(unseen, name)).all()
    assert scores.confounders.shape == (X.height, 3)


def test_baseline_is_standardized_and_exposes_targeted_selection_signal():
    X = ToyRealDataset().load()[0]
    _, scores = _fit_causal_score_map(X, np.random.default_rng(8))
    surface = _fit_baseline_surface(scores, np.random.default_rng(9))
    mean, signal = surface.evaluate(scores)

    np.testing.assert_allclose(mean.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(mean.std(ddof=0), 1.0, atol=1e-12)
    assert mean.shape == signal.shape == (X.height,)


def test_score_map_rejects_changed_schema_and_all_missing_numeric_column():
    X = ToyRealDataset().load()[0]
    score_map, _ = _fit_causal_score_map(X, np.random.default_rng(2))

    with pytest.raises(ValueError, match="columns"):
        score_map.transform(X.drop("income"))
    with pytest.raises(ValueError, match="finite observed value"):
        _fit_causal_score_map(
            X.with_columns(pl.lit(None, dtype=pl.Float64).alias("income")),
            np.random.default_rng(2),
        )
```

- [ ] **Step 2: Run the helper tests and verify the missing symbol failure**

Run: `uv run pytest tests/datasets/test_semi_synthetic_helpers.py -q`

Expected: collection fails because `_fit_causal_score_map` and `_fit_baseline_surface` are absent.

- [ ] **Step 3: Implement the frozen encoder, nonlinear library, role projections, and baseline surface**

Use these exact state shapes and defaults:

```python
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class _CausalScores:
    confounders: np.ndarray       # (n, 3)
    prognostic: np.ndarray        # (n, 2)
    treatment_only: np.ndarray    # (n, 2)
    modifiers: np.ndarray         # (n, 2)


@dataclass
class _CausalScoreMap:
    columns: tuple[str, ...]
    encoder: ColumnTransformer
    pair_indices: np.ndarray
    projections: dict[str, np.ndarray]
    means: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]

    def transform(self, X):
        frame = _covariates_to_pandas(X, self.columns)
        encoded = np.asarray(self.encoder.transform(frame), dtype=float)
        library = _nonlinear_library(encoded, self.pair_indices)
        return _CausalScores(
            **{
                name: (library @ self.projections[name] - self.means[name])
                / self.scales[name]
                for name in self.projections
            }
        )


@dataclass(frozen=True)
class _BaselineSurface:
    confounder_coefficients: np.ndarray
    prognostic_coefficients: np.ndarray
    mean: float
    scale: float
    signal_mean: float
    signal_scale: float

    def evaluate(self, scores):
        confounding = scores.confounders @ self.confounder_coefficients
        prognostic = scores.prognostic @ self.prognostic_coefficients
        mu0 = (confounding + prognostic - self.mean) / self.scale
        signal = (confounding - self.signal_mean) / self.signal_scale
        return mu0, signal
```

Implement `_covariates_to_pandas` so named frames must match the frozen columns exactly and NumPy inputs must match their width. Before fitting, reject numeric columns with infinite values or without a finite observed value. Convert categorical columns to object dtype, then use:

```python
numeric_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="__missing__"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
)
```

Build `_nonlinear_library` as `encoded`, `encoded**2`, `np.sin(encoded)`, `(encoded > 0).astype(float)`, and up to 16 frozen pairwise products. For each role, select at most five library columns per output score, draw Normal coefficients, and standardize the resulting scores on source `X`; replace zero scales with one. `_fit_baseline_surface` draws scaled Normal coefficients, standardizes total baseline risk, and separately standardizes its confounding contribution.

- [ ] **Step 4: Run helper and base tests**

Run: `uv run pytest tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_base.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the causal score helpers**

```bash
git add src/skcausal/datasets/semi_synthetic.py tests/datasets/test_semi_synthetic_helpers.py
git commit -m "feat: add frozen causal role scores"
```

### Task 3: Shared probability and confounding-calibration numerics

**Files:**
- Modify: `src/skcausal/datasets/semi_synthetic.py`
- Modify: `tests/datasets/test_semi_synthetic_helpers.py`

**Interfaces:**
- Consumes: arrays of logits/log-densities and scalar metric callables.
- Produces: `_softmax`, `_calibrate_softmax_intercepts`, `_normal_logpdf`, `_log_mixture`, `_normalized_log_weights`, and `_calibrate_confounding_strength`.

- [ ] **Step 1: Add failing numerical tests**

```python
from skcausal.datasets.semi_synthetic import (
    _calibrate_confounding_strength,
    _calibrate_softmax_intercepts,
    _log_mixture,
    _normal_logpdf,
    _normalized_log_weights,
    _softmax,
)


def test_softmax_intercepts_match_requested_empirical_marginal():
    logits = np.array(
        [[-2.0, 0.0, 1.0], [-1.0, 0.5, 0.0], [2.0, -1.0, 0.5], [1.0, 1.5, -2.0]]
    )
    target = np.array([0.2, 0.3, 0.5])
    intercepts = _calibrate_softmax_intercepts(logits, target)
    probabilities = _softmax(logits + intercepts)

    np.testing.assert_allclose(probabilities.mean(axis=0), target, atol=1e-9)


def test_log_mixture_and_normalized_weights_are_stable():
    x = np.array([-20.0, 0.0, 20.0])
    left = _normal_logpdf(x, np.zeros(3), 1.0)
    right = _normal_logpdf(x, np.ones(3), 2.0)
    mixed = _log_mixture(left, right, weight=0.25)
    weights = _normalized_log_weights(mixed)

    np.testing.assert_allclose(
        np.exp(mixed), 0.75 * np.exp(left) + 0.25 * np.exp(right)
    )
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_strength_calibration_finds_crossing_and_rejects_unreachable_target():
    strength, achieved = _calibrate_confounding_strength(
        metric=lambda value: value**2,
        target=0.49,
        initial_strength=0.5,
    )
    np.testing.assert_allclose(strength, 0.7, atol=1e-6)
    np.testing.assert_allclose(achieved, 0.49, atol=1e-6)

    with pytest.raises(ValueError, match="attainable"):
        _calibrate_confounding_strength(
            metric=lambda value: min(value, 0.2),
            target=0.5,
            initial_strength=1.0,
        )
```

- [ ] **Step 2: Run the numerical tests and verify the missing symbol failure**

Run: `uv run pytest tests/datasets/test_semi_synthetic_helpers.py -q`

Expected: collection fails on missing numerical helper imports.

- [ ] **Step 3: Implement stable numerical helpers**

```python
_LOG_2PI = np.log(2.0 * np.pi)


def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _normal_logpdf(value, location, scale):
    return -0.5 * ((value - location) / scale) ** 2 - np.log(scale) - 0.5 * _LOG_2PI


def _log_mixture(left, right, weight):
    if weight == 0.0:
        return left
    if weight == 1.0:
        return right
    return np.logaddexp(np.log1p(-weight) + left, np.log(weight) + right)


def _normalized_log_weights(log_values):
    shifted = log_values - np.max(log_values)
    weights = np.exp(shifted)
    return weights / weights.sum()
```

Implement `_calibrate_softmax_intercepts` with the final category fixed as the reference intercept and a damped Newton solve on the first `K - 1` mean-probability residuals. Stop when the maximum residual is below `1e-10`; raise `ValueError` after 100 iterations or on singular/nonfinite updates.

Implement `_calibrate_confounding_strength` with lower bound zero, an initial upper bound of `max(1.0, initial_strength)`, doubling up to 64 until the metric crosses the target, and bisection until both the metric error is below `1e-6` and the bracket width is below `1e-6`. Report the evaluated attainable interval in failures.

- [ ] **Step 4: Run all shared-helper tests**

Run: `uv run pytest tests/datasets/test_semi_synthetic_helpers.py -q`

Expected: all helper tests pass.

- [ ] **Step 5: Commit the numerical helpers**

```bash
git add src/skcausal/datasets/semi_synthetic.py tests/datasets/test_semi_synthetic_helpers.py
git commit -m "feat: add semi-synthetic calibration numerics"
```

### Task 4: Categorical assignment and Gaussian outcome DGP

**Files:**
- Create: `src/skcausal/datasets/semi_synthetic_categorical.py`
- Create: `tests/datasets/test_semi_synthetic_categorical_dgp.py`

**Interfaces:**
- Consumes: `BaseSemiSyntheticDataset`, frozen score/baseline helpers, softmax and intercept calibration helpers.
- Produces: `CategoricalSemiSyntheticDataset` with `column_types = {"t": "categorical"}`, string `treatment_levels_`, exact categorical `_log_prob`, `get_levels`, nonlinear oracle outcomes, Gaussian samples, and `get_test_params`.

- [ ] **Step 1: Write failing categorical DGP tests**

```python
# tests/datasets/test_semi_synthetic_categorical_dgp.py
import numpy as np
import polars as pl
import pytest

from skcausal.datasets.semi_synthetic_categorical import (
    CategoricalSemiSyntheticDataset,
)
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


def _all_level_probabilities(dataset, X):
    columns = []
    for level in dataset.treatment_levels_:
        treatment = pl.DataFrame({"t": [level] * X.height})
        columns.append(np.exp(dataset.log_prob(X, treatment)[:, 0]))
    return np.column_stack(columns)


def test_categorical_dgp_preserves_X_and_has_exact_calibrated_probabilities():
    target = np.array([0.2, 0.3, 0.5])
    source = ToyRealDataset(include_missing=True)
    dataset = CategoricalSemiSyntheticDataset(
        source,
        n_treatments=3,
        target_probabilities=target,
        randomized_weight=0.15,
        random_state=7,
    )
    X, treatment, outcome = dataset.load()
    probabilities = _all_level_probabilities(dataset, X)

    assert X.equals(source.load()[0])
    assert treatment.schema["t"] == pl.Categorical
    assert treatment.get_column("t").cast(pl.Utf8).is_in(dataset.treatment_levels_).all()
    assert outcome.schema["y"] == pl.Float64
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(probabilities.mean(axis=0), target, atol=1e-9)


def test_categorical_randomized_component_is_independent_of_X():
    dataset = CategoricalSemiSyntheticDataset(
        ToyRealDataset(),
        n_treatments=3,
        target_probabilities=[0.25, 0.5, 0.25],
        randomized_weight=1.0,
        random_state=4,
    )
    X, _, _ = dataset.load()
    probabilities = _all_level_probabilities(dataset, X)

    np.testing.assert_allclose(
        probabilities,
        np.tile([0.25, 0.5, 0.25], (X.height, 1)),
        atol=1e-12,
    )


def test_categorical_outcome_oracle_accepts_all_backends():
    dataset = CategoricalSemiSyntheticDataset(
        ToyRealDataset(), outcome_noise_scale=0.0, random_state=12
    )
    X, treatment, outcome = dataset.load()

    polars_mean = dataset.predict_y(X, treatment)
    pandas_mean = dataset.predict_y(X.to_pandas(), treatment.to_pandas())
    numpy_mean = dataset.predict_y(X.to_numpy(), treatment.to_numpy())

    np.testing.assert_allclose(polars_mean, outcome.to_numpy())
    np.testing.assert_allclose(polars_mean, pandas_mean)
    np.testing.assert_allclose(polars_mean, numpy_mean)
    assert dataset.get_levels().shape == (2, 1)


def test_categorical_dgp_validates_configuration_and_unknown_levels():
    with pytest.raises(ValueError, match="at least 2"):
        CategoricalSemiSyntheticDataset(ToyRealDataset(), n_treatments=1)
    with pytest.raises(ValueError, match="sum to one"):
        CategoricalSemiSyntheticDataset(
            ToyRealDataset(), n_treatments=2, target_probabilities=[0.8, 0.8]
        )

    dataset = CategoricalSemiSyntheticDataset(ToyRealDataset(), random_state=2)
    X, _, _ = dataset.load()
    unknown = pl.DataFrame({"t": ["unsupported"] * X.height})
    assert np.isneginf(dataset.log_prob(X, unknown)).all()
```

- [ ] **Step 2: Run the categorical tests and verify the missing module failure**

Run: `uv run pytest tests/datasets/test_semi_synthetic_categorical_dgp.py -q`

Expected: collection fails with `ModuleNotFoundError` for `semi_synthetic_categorical`.

- [ ] **Step 3: Implement categorical configuration and structural preparation**

Implement this constructor, setting every argument as an attribute before calling `super().__init__`:

```python
class CategoricalSemiSyntheticDataset(BaseSemiSyntheticDataset):
    column_types = {"t": "categorical"}

    def __init__(
        self,
        real_dataset,
        n_treatments=2,
        target_probabilities=None,
        confounding_strength=1.0,
        treatment_only_strength=1.0,
        randomized_weight=0.1,
        outcome_effect_scale=1.0,
        effect_heterogeneity_scale=0.5,
        outcome_noise_scale=1.0,
        random_state=42,
    ):
        if isinstance(n_treatments, bool) or not isinstance(n_treatments, int):
            raise TypeError("n_treatments must be an integer.")
        if n_treatments < 2:
            raise ValueError("n_treatments must be at least 2.")
        probabilities = (
            np.full(n_treatments, 1.0 / n_treatments)
            if target_probabilities is None
            else np.asarray(target_probabilities, dtype=float)
        )
        if (
            probabilities.shape != (n_treatments,)
            or np.any(probabilities <= 0.0)
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise ValueError(
                "target_probabilities must contain one positive value per treatment "
                "and sum to one."
            )
        for name, value in {
            "confounding_strength": confounding_strength,
            "treatment_only_strength": treatment_only_strength,
            "outcome_effect_scale": outcome_effect_scale,
            "effect_heterogeneity_scale": effect_heterogeneity_scale,
            "outcome_noise_scale": outcome_noise_scale,
        }.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        if not 0.0 <= float(randomized_weight) <= 1.0:
            raise ValueError("randomized_weight must be between 0 and 1.")
        self.n_treatments = n_treatments
        self.target_probabilities = target_probabilities
        self.confounding_strength = confounding_strength
        self.treatment_only_strength = treatment_only_strength
        self.randomized_weight = randomized_weight
        self.outcome_effect_scale = outcome_effect_scale
        self.effect_heterogeneity_scale = effect_heterogeneity_scale
        self.outcome_noise_scale = outcome_noise_scale
        self._validated_target_probabilities = probabilities
        super().__init__(real_dataset=real_dataset, random_state=random_state)
```

Validate `n_treatments >= 2`, strictly positive target probabilities summing to one, nonnegative strength/effect/noise scales, and `randomized_weight` in `[0, 1]`. In `_prepare_dgp`:

1. Fit `_CausalScoreMap` and `_BaselineSurface`, then store the source scores,
   source `mu0`, source targeted-selection signal, and
   `baseline_standard_deviation_ = np.std(source_mu0, ddof=0)`.
2. Set `treatment_levels_ = [str(index) for index in range(n_treatments)]`.
3. Draw category confounding loadings as centered `np.linspace(-1.0, 1.0, n_treatments)`.
4. Draw a `(2, n_treatments)` treatment-only coefficient matrix scaled by `1 / sqrt(2)` and center its category columns.
5. Draw `K - 1` outcome main-effect coefficients and a `(K - 1, 2)` modifier matrix using the configured scales.
6. Set `confounding_strength_ = confounding_strength` and calibrate the softmax intercepts at that strength.

- [ ] **Step 4: Implement categorical probabilities, sampling, log-probability, and outcome hooks**

Use the frozen baseline's targeted-selection signal in category-specific logits:

```python
def _confounded_logits(self, X, strength):
    scores = self.score_map_.transform(X)
    _, signal = self.baseline_surface_.evaluate(scores)
    treatment_only = scores.treatment_only @ self.treatment_only_coefficients_
    return (
        strength * signal[:, None] * self.confounding_loadings_[None, :]
        + self.treatment_only_strength * treatment_only
    )


def _probabilities(self, X, strength=None):
    strength = self.confounding_strength_ if strength is None else strength
    confounded = _softmax(
        self._confounded_logits(X, strength) + self.assignment_intercepts_
    )
    return (
        (1.0 - self.randomized_weight) * confounded
        + self.randomized_weight * self.target_probabilities_
    )
```

Sample categories by inverse row-wise CDF. Convert outputs to strings before `_to_polars` enforces categorical dtype. Parse categorical, pandas, and NumPy treatment inputs as strings and map them to category indices. Return `-inf` for strings outside `treatment_levels_`.

Construct category treatment bases with the first level as reference. `_predict_y` returns `mu0 + basis @ beta + einsum("ij,ij->i", basis @ gamma, q_M)`. `_sample_outcome` adds Gaussian noise with `outcome_noise_scale`. `_get_grid` delegates to `get_levels`, and `get_levels` returns the canonical categorical treatment frame.

Add `get_test_params` using `KangSchaferContinuous(n=64, random_state=3)` as a lightweight `BaseDataset` source.

- [ ] **Step 5: Run categorical, base, and helper tests**

Run: `uv run pytest tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit the categorical reference DGP**

```bash
git add src/skcausal/datasets/semi_synthetic_categorical.py tests/datasets/test_semi_synthetic_categorical_dgp.py
git commit -m "feat: add categorical semi-synthetic DGP"
```

### Task 5: Categorical oracle-bias calibration

**Files:**
- Modify: `src/skcausal/datasets/semi_synthetic_categorical.py`
- Modify: `tests/datasets/test_semi_synthetic_categorical_dgp.py`

**Interfaces:**
- Consumes: fixed category outcome means, candidate category probabilities, `_calibrate_confounding_strength`.
- Produces: optional `target_confounding_bias`, plus fitted `confounding_bias_` and `confounding_bias_ratio_`.

- [ ] **Step 1: Add failing fixed-strength and target-calibration tests**

```python
def test_categorical_target_bias_calibrates_strength_without_changing_marginal():
    dataset = CategoricalSemiSyntheticDataset(
        ToyRealDataset(),
        n_treatments=3,
        target_probabilities=[0.3, 0.4, 0.3],
        confounding_strength=1.0,
        target_confounding_bias=0.35,
        treatment_only_strength=0.0,
        randomized_weight=0.1,
        random_state=21,
    )
    X, _, _ = dataset.load()
    probabilities = _all_level_probabilities(dataset, X)

    np.testing.assert_allclose(dataset.confounding_bias_ratio_, 0.35, atol=2e-3)
    np.testing.assert_allclose(
        probabilities.mean(axis=0), [0.3, 0.4, 0.3], atol=1e-9
    )
    assert dataset.confounding_strength_ >= 0.0


def test_fully_randomized_categorical_assignment_has_zero_oracle_bias():
    dataset = CategoricalSemiSyntheticDataset(
        ToyRealDataset(), randomized_weight=1.0, random_state=8
    )
    np.testing.assert_allclose(dataset.confounding_bias_ratio_, 0.0, atol=1e-12)
```

- [ ] **Step 2: Run the new tests and verify the constructor failure**

Run: `uv run pytest tests/datasets/test_semi_synthetic_categorical_dgp.py -q`

Expected: the target-calibration test fails because `target_confounding_bias` is not accepted.

- [ ] **Step 3: Implement categorical bias evaluation and calibration**

Add `target_confounding_bias=None` to the constructor, assign it to `self.target_confounding_bias` before `super().__init__`, and validate it as nonnegative when present. For each category, calculate the source-population outcome mean and the probability-weighted observational mean. Define raw bias as the range of category-specific distortions and normalized bias as raw bias divided by `baseline_standard_deviation_`, the empirical standard deviation of the frozen source `mu0`.

Refactor probability calculation so candidate strengths receive candidate-specific calibrated intercepts without mutating final state:

```python
def _probabilities_at_strength(self, X, strength):
    logits = self._confounded_logits(X, strength)
    intercepts = _calibrate_softmax_intercepts(logits, self.target_probabilities_)
    confounded = _softmax(logits + intercepts)
    probabilities = (
        (1.0 - self.randomized_weight) * confounded
        + self.randomized_weight * self.target_probabilities_
    )
    return probabilities, intercepts
```

If a target is provided, call `_calibrate_confounding_strength` with the normalized bias metric while all score/outcome arrays remain fixed. Store the final intercepts and fitted bias attributes. If no target is provided, evaluate and store the attributes at the configured fixed strength. Special-case `randomized_weight == 1` so the final bias is exactly zero and an unreachable positive target raises a clear `ValueError`.

- [ ] **Step 4: Run categorical tests**

Run: `uv run pytest tests/datasets/test_semi_synthetic_categorical_dgp.py -q`

Expected: all categorical tests pass.

- [ ] **Step 5: Commit categorical calibration**

```bash
git add src/skcausal/datasets/semi_synthetic_categorical.py tests/datasets/test_semi_synthetic_categorical_dgp.py
git commit -m "feat: calibrate categorical confounding bias"
```

### Task 6: Scalar-continuous transformed assignment and Gaussian outcome DGP

**Files:**
- Create: `src/skcausal/datasets/semi_synthetic_continuous.py`
- Create: `tests/datasets/test_semi_synthetic_continuous_dgp.py`

**Interfaces:**
- Consumes: `BaseSemiSyntheticDataset`, score/baseline helpers, normal log-density and log-mixture helpers.
- Produces: `ContinuousSemiSyntheticDataset` supporting `"real"`, `"positive"`, and `(lower, upper)` domains; exact transformed mixture density; nonlinear treatment basis; Gaussian outcome; and `get_test_params`.

- [ ] **Step 1: Write failing domain, density, and outcome tests**

```python
# tests/datasets/test_semi_synthetic_continuous_dgp.py
import numpy as np
import pytest

from skcausal.datasets.semi_synthetic_continuous import (
    ContinuousSemiSyntheticDataset,
)
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


def _integrated_density(dataset, X_row, grid):
    repeated_X = X_row[np.zeros(grid.size, dtype=int)]
    treatment = grid.reshape(-1, 1)
    density = np.exp(dataset.log_prob(repeated_X, treatment)[:, 0])
    return np.trapezoid(density, grid)


def test_real_continuous_density_integrates_and_outcome_oracle_matches_noise_free_Y():
    dataset = ContinuousSemiSyntheticDataset(
        ToyRealDataset(),
        treatment_domain="real",
        outcome_noise_scale=0.0,
        random_state=5,
    )
    X, treatment, outcome = dataset.load()
    grid = np.linspace(-10.0, 10.0, 20001)

    np.testing.assert_allclose(
        _integrated_density(dataset, X.to_numpy()[:1], grid), 1.0, atol=2e-3
    )
    np.testing.assert_allclose(dataset.predict_y(X, treatment), outcome.to_numpy())
    assert dataset.get_grid(11).shape == (11, 1)


def test_positive_and_bounded_domains_have_exact_support_and_normalized_density():
    positive = ContinuousSemiSyntheticDataset(
        ToyRealDataset(), treatment_domain="positive", random_state=6
    )
    bounded = ContinuousSemiSyntheticDataset(
        ToyRealDataset(), treatment_domain=(-2.0, 3.0), random_state=7
    )
    X_positive, treatment_positive, _ = positive.load()
    X_bounded, treatment_bounded, _ = bounded.load()

    assert (treatment_positive.get_column("t") > 0).all()
    assert (treatment_bounded.get_column("t") > -2.0).all()
    assert (treatment_bounded.get_column("t") < 3.0).all()
    assert np.isneginf(
        positive.log_prob(X_positive.head(1), np.array([[0.0]]))[0, 0]
    )
    assert np.isneginf(
        bounded.log_prob(X_bounded.head(1), np.array([[3.0]]))[0, 0]
    )

    positive_grid = np.geomspace(1e-5, 100.0, 30000)
    bounded_grid = np.linspace(-2.0 + 1e-6, 3.0 - 1e-6, 30000)
    np.testing.assert_allclose(
        _integrated_density(positive, X_positive.to_numpy()[:1], positive_grid),
        1.0,
        atol=4e-3,
    )
    np.testing.assert_allclose(
        _integrated_density(bounded, X_bounded.to_numpy()[:1], bounded_grid),
        1.0,
        atol=4e-3,
    )


def test_continuous_randomized_assignment_density_is_independent_of_X():
    dataset = ContinuousSemiSyntheticDataset(
        ToyRealDataset(), randomized_weight=1.0, random_state=10
    )
    X, _, _ = dataset.load()
    treatment = np.full((X.height, 1), 0.25)
    log_density = dataset.log_prob(X, treatment)

    np.testing.assert_allclose(log_density, np.repeat(log_density[:1], X.height, axis=0))


def test_continuous_dgp_rejects_invalid_domains_and_noise():
    with pytest.raises(ValueError, match="treatment_domain"):
        ContinuousSemiSyntheticDataset(
            ToyRealDataset(), treatment_domain=(2.0, -1.0)
        )
    with pytest.raises(ValueError, match="treatment_noise_scale"):
        ContinuousSemiSyntheticDataset(
            ToyRealDataset(), treatment_noise_scale=0.0
        )
```

- [ ] **Step 2: Run the continuous tests and verify the missing module failure**

Run: `uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py -q`

Expected: collection fails with `ModuleNotFoundError` for `semi_synthetic_continuous`.

- [ ] **Step 3: Implement domain transforms and continuous structural preparation**

Implement this constructor and validation:

```python
class ContinuousSemiSyntheticDataset(BaseSemiSyntheticDataset):
    column_types = {"t": "continuous"}

    def __init__(
        self,
        real_dataset,
        treatment_domain="real",
        confounding_strength=1.0,
        treatment_only_strength=1.0,
        randomized_weight=0.1,
        treatment_noise_scale=1.0,
        outcome_effect_scale=1.0,
        effect_heterogeneity_scale=0.5,
        outcome_noise_scale=1.0,
        random_state=42,
    ):
        if isinstance(treatment_domain, str):
            if treatment_domain not in {"real", "positive"}:
                raise ValueError(
                    "treatment_domain must be 'real', 'positive', or a finite "
                    "(lower, upper) tuple."
                )
        elif (
            not isinstance(treatment_domain, tuple)
            or len(treatment_domain) != 2
            or not np.issubdtype(np.asarray(treatment_domain).dtype, np.number)
            or not np.isfinite(treatment_domain).all()
            or not float(treatment_domain[0]) < float(treatment_domain[1])
        ):
            raise ValueError(
                "treatment_domain must be 'real', 'positive', or a finite "
                "(lower, upper) tuple."
            )
        if float(treatment_noise_scale) <= 0.0:
            raise ValueError("treatment_noise_scale must be positive.")
        for name, value in {
            "confounding_strength": confounding_strength,
            "treatment_only_strength": treatment_only_strength,
            "outcome_effect_scale": outcome_effect_scale,
            "effect_heterogeneity_scale": effect_heterogeneity_scale,
            "outcome_noise_scale": outcome_noise_scale,
        }.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        if not 0.0 <= float(randomized_weight) <= 1.0:
            raise ValueError("randomized_weight must be between 0 and 1.")
        self.treatment_domain = treatment_domain
        self.confounding_strength = confounding_strength
        self.treatment_only_strength = treatment_only_strength
        self.randomized_weight = randomized_weight
        self.treatment_noise_scale = treatment_noise_scale
        self.outcome_effect_scale = outcome_effect_scale
        self.effect_heterogeneity_scale = effect_heterogeneity_scale
        self.outcome_noise_scale = outcome_noise_scale
        super().__init__(real_dataset=real_dataset, random_state=random_state)
```

Accept only `"real"`, `"positive"`, or a two-finite-number tuple with lower `<` upper. Require positive treatment noise, nonnegative remaining scales, and mixture weight in `[0, 1]`.

In `_prepare_dgp`, fit scores/baseline; store the source scores, source `mu0`, source targeted-selection signal, and `baseline_standard_deviation_ = np.std(source_mu0, ddof=0)`; draw a length-two treatment-only coefficient vector scaled by `1 / sqrt(2)`; draw the five treatment-basis main-effect coefficients and `(5, 2)` modifier matrix; and set `confounding_strength_`. Define the central latent intervention region using `NormalDist().inv_cdf(0.01)` and `NormalDist().inv_cdf(0.99)` times `treatment_noise_scale`. Freeze the raw treatment-basis values, source-grid means/scales, and hinge knots at latent quantiles 1/3 and 2/3; subtract the basis at latent reference zero so the reference effect is zero.

Implement `_forward_transform`, `_inverse_transform`, and `_log_abs_inverse_jacobian` for:

```python
# real
a = z
log_jacobian = 0.0

# positive
a = np.exp(z)
z = np.log(a)
log_jacobian = -np.log(a)

# bounded
unit = (a - lower) / (upper - lower)
z = np.log(unit) - np.log1p(-unit)
log_jacobian = -np.log(upper - lower) - np.log(unit) - np.log1p(-unit)
```

- [ ] **Step 4: Implement sampling, exact density, treatment basis, and outcomes**

The paired latent confounded mean is:

```python
def _latent_mean(self, X, strength=None):
    strength = self.confounding_strength_ if strength is None else strength
    scores = self.score_map_.transform(X)
    _, signal = self.baseline_surface_.evaluate(scores)
    treatment_only = scores.treatment_only @ self.treatment_only_coefficients_
    return strength * signal + self.treatment_only_strength * treatment_only
```

For sampling, draw a Bernoulli mixture indicator; use location zero for randomized rows and `_latent_mean` for confounded rows; then add Gaussian noise with `treatment_noise_scale` and apply the configured transform.

For `_log_prob`, inverse-transform valid treatments, compute confounded and randomized latent Normal log-densities, combine them with `_log_mixture`, and add the inverse Jacobian. Assign `-inf` outside positive/bounded support.

Evaluate the treatment basis from latent `z = T^{-1}(a)` using linear, quadratic, sinusoidal, and two hinge terms; apply frozen centering/scaling and subtract the transformed reference basis. `_predict_y` combines baseline, basis main effects, and modifier interactions. `_sample_outcome` adds configured Gaussian noise. `_get_grid` linearly spaces latent points over the frozen central region before applying the domain transform.

Add `get_test_params` using `KangSchaferContinuous(n=64, random_state=4)` and include real and bounded configurations.

- [ ] **Step 5: Run continuous, base, and helper tests**

Run: `uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit the continuous reference DGP**

```bash
git add src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
git commit -m "feat: add continuous semi-synthetic DGP"
```

### Task 7: Continuous oracle-bias calibration

**Files:**
- Modify: `src/skcausal/datasets/semi_synthetic_continuous.py`
- Modify: `tests/datasets/test_semi_synthetic_continuous_dgp.py`

**Interfaces:**
- Consumes: fixed grid outcome means, candidate conditional continuous densities, `_normalized_log_weights`, `_calibrate_confounding_strength`.
- Produces: optional `target_confounding_bias`, plus fitted `confounding_bias_` and `confounding_bias_ratio_`.

- [ ] **Step 1: Add failing continuous bias calibration and randomized sanity tests**

```python
def test_continuous_target_bias_calibrates_strength():
    dataset = ContinuousSemiSyntheticDataset(
        ToyRealDataset(),
        target_confounding_bias=0.25,
        treatment_only_strength=0.0,
        randomized_weight=0.15,
        treatment_noise_scale=1.2,
        random_state=18,
    )

    np.testing.assert_allclose(dataset.confounding_bias_ratio_, 0.25, atol=2e-3)
    assert dataset.confounding_strength_ >= 0.0
    assert dataset.confounding_bias_ >= 0.0


def test_fully_randomized_continuous_assignment_has_zero_oracle_bias():
    dataset = ContinuousSemiSyntheticDataset(
        ToyRealDataset(), randomized_weight=1.0, random_state=20
    )
    np.testing.assert_allclose(dataset.confounding_bias_ratio_, 0.0, atol=1e-12)
```

- [ ] **Step 2: Run the new tests and verify the constructor failure**

Run: `uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py -q`

Expected: the calibration test fails because `target_confounding_bias` is not accepted.

- [ ] **Step 3: Implement continuous finite-population bias evaluation**

Add `target_confounding_bias=None` to the constructor, assign it to `self.target_confounding_bias` before `super().__init__`, and validate it as nonnegative. At each row of `get_grid(31)`:

1. Repeat the intervention across source `X`.
2. Evaluate `_predict_y` to obtain `m(a, X_i)`.
3. Evaluate candidate-strength log-density without mutating `confounding_strength_`.
4. Normalize density weights with `_normalized_log_weights`.
5. Subtract the unweighted mean outcome from the density-weighted mean.

Define raw continuous bias as the RMS of these 31 distortions and divide by the baseline risk standard deviation for the normalized ratio. Use `_calibrate_confounding_strength` when a target is present, then store `confounding_strength_`, `confounding_bias_`, and `confounding_bias_ratio_`. At fixed strength, evaluate and store the same attributes. A fully randomized mixture must return exact zero by construction.

- [ ] **Step 4: Run continuous tests**

Run: `uv run pytest tests/datasets/test_semi_synthetic_continuous_dgp.py -q`

Expected: all continuous tests pass.

- [ ] **Step 5: Commit continuous calibration**

```bash
git add src/skcausal/datasets/semi_synthetic_continuous.py tests/datasets/test_semi_synthetic_continuous_dgp.py
git commit -m "feat: calibrate continuous confounding bias"
```

### Task 8: Public exports, discovery, and full verification

**Files:**
- Modify: `src/skcausal/datasets/__init__.py`
- Modify: `tests/test_lookup.py`

**Interfaces:**
- Consumes: the three implemented dataset classes and their `get_test_params` methods.
- Produces: stable imports from `skcausal.datasets`, successful `all_datasets` discovery, formatting-clean source, and a verified full suite.

- [ ] **Step 1: Add the failing public import and discovery test**

```python
# tests/test_lookup.py
from skcausal.datasets import (
    BaseSemiSyntheticDataset,
    CategoricalSemiSyntheticDataset,
    ContinuousSemiSyntheticDataset,
)


def test_semi_synthetic_datasets_are_public_and_discoverable():
    discovered = all_datasets(return_names=True)
    names = {name for name, _ in discovered}

    assert BaseSemiSyntheticDataset.__name__ == "BaseSemiSyntheticDataset"
    assert "CategoricalSemiSyntheticDataset" in names
    assert "ContinuousSemiSyntheticDataset" in names
```

- [ ] **Step 2: Run the lookup test and verify the missing export failure**

Run: `uv run pytest tests/test_lookup.py::test_semi_synthetic_datasets_are_public_and_discoverable -q`

Expected: collection fails because the new classes are not exported from `skcausal.datasets`.

- [ ] **Step 3: Export the new public classes**

```python
# src/skcausal/datasets/__init__.py
from .semi_synthetic import BaseSemiSyntheticDataset
from .semi_synthetic_categorical import CategoricalSemiSyntheticDataset
from .semi_synthetic_continuous import ContinuousSemiSyntheticDataset
```

Add all three names to `__all__` without changing existing exports.

- [ ] **Step 4: Run focused integration and discovery tests**

Run: `uv run pytest tests/test_lookup.py tests/datasets/test_all_datasets.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_continuous_dgp.py -q`

Expected: all selected tests pass, including `QuickTester` cloning, backend, deterministic-random-state, and treatment-column-type checks for both concrete DGPs.

- [ ] **Step 5: Format and lint changed Python files**

Run: `uv run ruff format src/skcausal/datasets/semi_synthetic.py src/skcausal/datasets/semi_synthetic_categorical.py src/skcausal/datasets/semi_synthetic_continuous.py src/skcausal/datasets/__init__.py tests/datasets/__init__.py tests/datasets/_semi_synthetic_test_utils.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_continuous_dgp.py tests/test_lookup.py`

Expected: Ruff reports formatted files without an error.

Run: `uv run ruff check src/skcausal/datasets/semi_synthetic.py src/skcausal/datasets/semi_synthetic_categorical.py src/skcausal/datasets/semi_synthetic_continuous.py src/skcausal/datasets/__init__.py tests/datasets/__init__.py tests/datasets/_semi_synthetic_test_utils.py tests/datasets/test_semi_synthetic_base.py tests/datasets/test_semi_synthetic_helpers.py tests/datasets/test_semi_synthetic_categorical_dgp.py tests/datasets/test_semi_synthetic_continuous_dgp.py tests/test_lookup.py`

Expected: `All checks passed!`

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -q`

Expected: the complete repository test suite passes with no failures.

- [ ] **Step 7: Verify implemented public signatures against the approved contract**

Run: `uv run python -c 'import inspect; from skcausal.datasets import BaseSemiSyntheticDataset, CategoricalSemiSyntheticDataset, ContinuousSemiSyntheticDataset; print(inspect.signature(BaseSemiSyntheticDataset.log_prob)); print(inspect.signature(BaseSemiSyntheticDataset.sample)); print(inspect.signature(CategoricalSemiSyntheticDataset)); print(inspect.signature(ContinuousSemiSyntheticDataset))'`

Expected: `log_prob` accepts `(self, X, treatment)`, `sample` accepts `(self, random_state)`, both concrete constructors begin with `real_dataset`, and both expose the approved confounding, treatment-only, randomized-mixture, treatment/outcome-noise, effect, and random-state parameters.

- [ ] **Step 8: Commit public integration updates**

```bash
git add src/skcausal/datasets/__init__.py tests/test_lookup.py
git commit -m "feat: expose semi-synthetic observational DGPs"
```

### Task 9: Final regression verification

**Files:**
- Verify only; modify implementation or tests only if a concrete regression is found.

**Interfaces:**
- Consumes: the completed public dataset implementation.
- Produces: evidence that legacy semi-synthetic datasets and the full repository remain correct.

- [ ] **Step 1: Run legacy semi-synthetic regression tests explicitly**

Run: `uv run pytest tests/datasets/test_semi_synthetic_regressor.py tests/datasets/test_semi_synthetic_classifier.py -q`

Expected: all legacy semi-synthetic tests pass unchanged.

- [ ] **Step 2: Run the complete dataset and datatype suites**

Run: `uv run pytest tests/datasets tests/datatypes -q`

Expected: all dataset and datatype tests pass.

- [ ] **Step 3: Run the complete repository suite once more from a clean command**

Run: `uv run pytest -q`

Expected: the full suite passes with no failures.

- [ ] **Step 4: Confirm a clean worktree and review the implementation diff**

Run: `git status --short`

Expected: no uncommitted files.

Run: `git log --oneline --decorate -10`

Expected: the task commits appear in dependency order after the approved design and implementation-plan commits.
