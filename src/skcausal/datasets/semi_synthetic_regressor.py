from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

from skcausal.datasets.base import BaseSyntheticDataset

__all__ = ["SemiSyntheticRegressor"]


def _default_test_dataset() -> tuple[np.ndarray, np.ndarray]:
    return make_regression(
        n_samples=2000,
        n_features=5,
        noise=5.0,
        random_state=17,
    )


def _as_feature_matrix(values, *, expected_width: int | None, name: str):
    columns = None

    if isinstance(values, pl.DataFrame):
        columns = list(values.columns)
        values = values.to_numpy()
    elif isinstance(values, pd.DataFrame):
        columns = list(values.columns)
        values = values.to_numpy()

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if expected_width is None or expected_width == 1:
            array = array.reshape(-1, 1)
        elif array.size == expected_width:
            array = array.reshape(1, -1)
        else:
            raise ValueError(
                f"{name} expects inputs with shape (n_samples, {expected_width})."
            )

    if array.ndim != 2:
        raise ValueError(f"{name} expects a 2D array-like input.")

    if expected_width is not None and array.shape[1] != expected_width:
        raise ValueError(
            f"{name} expects inputs with shape (n_samples, {expected_width})."
        )

    return array, columns


def _as_target_array(values, *, name: str):
    if isinstance(values, pl.DataFrame):
        values = values.to_numpy()
    elif isinstance(values, pd.DataFrame):
        values = values.to_numpy()
    elif isinstance(values, pd.Series):
        values = values.to_numpy()

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _standardize_matrix(values: np.ndarray):
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    return (values - means) / scales, means, scales


def _standardize_vector(values: np.ndarray):
    mean = float(values.mean())
    scale = float(values.std(ddof=0))
    if scale == 0.0:
        scale = 1.0
    return (values - mean) / scale, mean, scale


class SemiSyntheticRegressor(BaseSyntheticDataset):
    r"""Semi-synthetic continuous-treatment dataset from a regression problem.

    The dataset starts from a supervised regression sample ``(X, y)`` returned by
    ``_load_dataset`` or by the optional ``load_dataset`` callable. The features
    and target are standardized, a clone of the supplied scikit-learn regressor
    is fit on the normalized regression problem, and the fitted predictions
    become the observed treatment values. The structural response then adds a
    random spline effect of the treatment to the fitted regression mean.

    If the raw regression sample is :math:`(X_i^{\mathrm{raw}}, y_i^{\mathrm{raw}})`
    for :math:`i = 1, \ldots, n`, the dataset first standardizes each feature
    column and the target:

    .. math::

        X_{ij} =
        \frac{X_{ij}^{\mathrm{raw}} - \mu_j}{s_j},
        \qquad
        y_i = \frac{y_i^{\mathrm{raw}} - \mu_y}{s_y},

    where zero empirical standard deviations are replaced by 1 in the code.

    A cloned regressor is then fit on the normalized regression task and its
    fitted predictions define the treatment:

    .. math::

        \hat{m} = \operatorname{fit}(\text{regressor}, X, y),
        \qquad
        T_i = \hat{m}(X_i).

    Let :math:`B(t) \in \mathbb{R}^K` denote the spline basis produced by
    ``SplineTransformer`` after fitting on the realized treatments
    :math:`T_1, \ldots, T_n`. The random spline coefficients are sampled as

    .. math::

        \beta_k \stackrel{\mathrm{iid}}{\sim} \mathcal{N}\!\left(0, \frac{1}{K}\right),
        \qquad k = 1, \ldots, K,

    and the treatment effect is centered over the observed treatment sample:

    .. math::

        g(t) = \lambda
        \left(
            B(t)^\top \beta
            - \frac{1}{n} \sum_{i=1}^n B(T_i)^\top \beta
        \right),

    where :math:`\lambda =` ``treatment_effect_scale``. If all realized
    treatments are identical, the implementation skips the spline fit and uses
    :math:`g(t) \equiv 0`.

    The noiseless response surface exposed by :meth:`predict_y` is therefore

    .. math::

        \mu(x, t) = \hat{m}(x) + g(t).

    The observed outcomes returned by :meth:`load` are sampled at the realized
    treatments with additive Gaussian noise:

    .. math::

        Y_i = \mu(X_i, T_i) + \varepsilon_i,
        \qquad
        \varepsilon_i \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2),

    where :math:`\sigma =` ``outcome_noise_scale`` is the noise standard
    deviation used in the implementation.
    """

    column_types = {"t": "continuous"}

    def __init__(
        self,
        regressor,
        load_dataset: Callable[[], tuple[object, object]] | None = None,
        random_state: int = 42,
        n_spline_knots: int = 6,
        spline_degree: int = 3,
        treatment_effect_scale: float = 1.0,
        outcome_noise_scale: float = 1.0,
    ):
        if n_spline_knots < 2:
            raise ValueError("n_spline_knots must be at least 2.")
        if spline_degree < 0:
            raise ValueError("spline_degree must be non-negative.")

        self.regressor = regressor
        self.load_dataset = load_dataset
        self.n_spline_knots = n_spline_knots
        self.spline_degree = spline_degree
        self.treatment_effect_scale = treatment_effect_scale
        self.outcome_noise_scale = outcome_noise_scale

        super().__init__(n=0, random_state=random_state)

        self.covariate_columns_ = None
        self.normalized_target_ = None
        self.regressor_ = None
        self.spline_transformer_ = None
        self.spline_coefficients_ = None
        self.treatment_effect_offset_ = 0.0
        self.treatment_range_ = None
        self._n_features = None

        self._prepare()

    def _load_dataset(self):
        if self.load_dataset is None:
            raise NotImplementedError(
                "SemiSyntheticRegressor requires a load_dataset callable or a "
                "subclass override of _load_dataset()."
            )
        return self.load_dataset()

    def _get_covariates(self) -> np.ndarray:
        loaded = self._load_dataset()
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise ValueError("_load_dataset must return exactly (X, y).")

        covariates, target = loaded
        covariate_array, columns = _as_feature_matrix(
            covariates,
            expected_width=None,
            name=f"{type(self).__name__} covariates",
        )
        target_array = _as_target_array(target, name=f"{type(self).__name__} target")

        if covariate_array.shape[0] != target_array.shape[0]:
            raise ValueError(
                "_load_dataset must return covariates and targets with the same "
                "number of rows."
            )

        normalized_covariates, _, _ = _standardize_matrix(covariate_array)
        normalized_target, _, _ = _standardize_vector(target_array)

        self.n = covariate_array.shape[0]
        self._n_features = covariate_array.shape[1]
        self.covariate_columns_ = columns or [
            f"x{i}" for i in range(covariate_array.shape[1])
        ]
        self.normalized_target_ = normalized_target

        return normalized_covariates

    def _clone_regressor(self):
        estimator = clone(self.regressor)
        params = estimator.get_params(deep=False)
        if "random_state" in params and params["random_state"] is None:
            estimator.set_params(random_state=self.random_state)
        return estimator

    def _fit_treatment_effect(self, treatments: np.ndarray):
        treatment_array, _ = _as_feature_matrix(
            treatments,
            expected_width=1,
            name=f"{type(self).__name__} treatments",
        )
        treatment_values = treatment_array[:, 0]
        treatment_min = float(np.min(treatment_values))
        treatment_max = float(np.max(treatment_values))
        self.treatment_range_ = (treatment_min, treatment_max)

        if np.isclose(treatment_min, treatment_max):
            self.spline_transformer_ = None
            self.spline_coefficients_ = np.zeros(1, dtype=float)
            self.treatment_effect_offset_ = 0.0
            return

        unique_count = np.unique(np.round(treatment_values, decimals=12)).size
        n_knots = min(self.n_spline_knots, max(2, unique_count))
        self.spline_transformer_ = SplineTransformer(
            n_knots=n_knots,
            degree=self.spline_degree,
            include_bias=False,
            extrapolation="continue",
        )
        basis = self.spline_transformer_.fit_transform(treatment_array)
        coefficient_scale = 1.0 / np.sqrt(max(1, basis.shape[1]))
        self.spline_coefficients_ = self._rng.normal(
            loc=0.0,
            scale=coefficient_scale,
            size=basis.shape[1],
        )
        self.treatment_effect_offset_ = float(
            (basis @ self.spline_coefficients_).mean()
        )

    def _evaluate_treatment_effect(self, treatments: np.ndarray) -> np.ndarray:
        treatment_array, _ = _as_feature_matrix(
            treatments,
            expected_width=1,
            name=f"{type(self).__name__} treatments",
        )

        if self.spline_transformer_ is None:
            return np.zeros(treatment_array.shape[0], dtype=float)

        basis = self.spline_transformer_.transform(treatment_array)
        effect = basis @ self.spline_coefficients_
        effect = effect - self.treatment_effect_offset_
        return self.treatment_effect_scale * effect

    def _get_treatments(self, covariates) -> np.ndarray:
        covariate_array, _ = _as_feature_matrix(
            covariates,
            expected_width=self._n_features,
            name=f"{type(self).__name__} covariates",
        )
        if self.normalized_target_ is None:
            raise RuntimeError("Covariates must be prepared before treatments.")

        self.regressor_ = self._clone_regressor()
        self.regressor_.fit(covariate_array, self.normalized_target_)

        treatments = np.asarray(self.regressor_.predict(covariate_array), dtype=float)
        treatments = treatments.reshape(-1, 1)
        self._fit_treatment_effect(treatments)
        return treatments

    def _inject_outcome_noise(self, expected_outcomes, covariates, treatments):
        expected = np.asarray(expected_outcomes, dtype=float).reshape(-1, 1)

        return expected + self._rng.normal(
            loc=0.0,
            scale=self.outcome_noise_scale,
            size=expected.shape[0],
        ).reshape(-1, 1)

    def _predict_y(self, covariates, treatments) -> np.ndarray:
        covariate_array, _ = _as_feature_matrix(
            covariates,
            expected_width=self._n_features,
            name=f"{type(self).__name__} covariates",
        )
        treatment_array, _ = _as_feature_matrix(
            treatments,
            expected_width=1,
            name=f"{type(self).__name__} treatments",
        )

        baseline = np.asarray(self.regressor_.predict(covariate_array), dtype=float)
        effect = self._evaluate_treatment_effect(treatment_array)
        return (baseline + effect).reshape(-1, 1)

    def _covariate_frame(self, covariates: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {
                column: covariates[:, idx]
                for idx, column in enumerate(self.covariate_columns_)
            }
        )

    def _treatment_frame(self, treatments: np.ndarray) -> pl.DataFrame:
        return self._to_polars(
            pl.DataFrame({"t": np.asarray(treatments, dtype=float).reshape(-1)})
        )

    def _outcome_frame(self, outcomes: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({"y": np.asarray(outcomes, dtype=float).reshape(-1)})

    def _prepare(self, n: int = None):
        if n is not None:
            raise ValueError(
                "SemiSyntheticRegressor derives its sample size from the loaded "
                "dataset and does not support prepare(n=...)."
            )

        covariates = self._get_covariates()
        treatments = self._get_treatments(covariates)
        outcomes = self._get_outcomes(covariates, treatments)

        self._covariates = self._covariate_frame(covariates)
        self._treatments = self._treatment_frame(treatments)
        self._outcomes = self._outcome_frame(outcomes)
        self.n = self._covariates.height

        return self

    def get_grid(self, n: int = 100) -> pl.DataFrame:
        if self.treatment_range_ is None:
            raise RuntimeError("The dataset must be prepared before requesting a grid.")

        lower, upper = self.treatment_range_
        if np.isclose(lower, upper):
            grid = np.repeat(lower, n)
        else:
            grid = np.linspace(lower, upper, n)

        return self._coerce_treatment_frame(pl.DataFrame({"t": grid}))

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        from sklearn.datasets import load_diabetes
        from functools import partial

        _load = partial(load_diabetes, return_X_y=True, as_frame=True)

        return [
            {
                "regressor": LinearRegression(),
                "load_dataset": _load,
                "treatment_effect_scale": 5,
                "random_state": 0,
            },
            {
                "regressor": LinearRegression(),
                "load_dataset": _default_test_dataset,
                "random_state": 1,
                "treatment_effect_scale": 0.5,
            },
        ]
