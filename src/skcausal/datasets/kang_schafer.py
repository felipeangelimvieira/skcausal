from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from skcausal.datasets.base import BaseSyntheticDataset

__all__ = [
    "KangSchaferBinary",
    "KangSchaferBinaryMisspecified",
    "KangSchaferContinuous",
    "KangSchaferContinuousMisspecified",
    "KangSchaferBinaryCrossValidation",
]

_LATENT_COLUMNS = ["x1", "x2", "x3", "x4"]
_TRANSFORMED_COLUMNS = ["z1", "z2", "z3", "z4"]
_PROPENSITY_WEIGHTS = np.array([1.0, -0.5, 0.25, 0.1], dtype=float)
_OUTCOME_WEIGHTS = np.array([27.4, 13.7, 13.7, 13.7], dtype=float)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped_values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped_values))


def _to_numpy_and_columns(values, *, expected_width: int, name: str):
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
        if expected_width == 1:
            array = array.reshape(-1, 1)
        elif array.size == expected_width:
            array = array.reshape(1, -1)

    if array.ndim != 2 or array.shape[1] != expected_width:
        raise ValueError(
            f"{name} expects inputs with shape (n_samples, {expected_width})."
        )

    return array, columns


def _latent_to_frame(latent_covariates: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame(
        {
            column: latent_covariates[:, idx]
            for idx, column in enumerate(_LATENT_COLUMNS)
        }
    )


def _transform_covariates(latent_covariates: np.ndarray) -> pl.DataFrame:
    x1 = latent_covariates[:, 0]
    x2 = latent_covariates[:, 1]
    x3 = latent_covariates[:, 2]
    x4 = latent_covariates[:, 3]
    return pl.DataFrame(
        {
            "z1": np.exp(x1 / 2.0),
            "z2": x2 / (1.0 + np.exp(x1)) + 10.0,
            "z3": ((x1 * x3) / 25.0 + 0.6) ** 3,
            "z4": (x2 + x4 + 20.0) ** 2,
        }
    )


class _BaseKangSchafer(BaseSyntheticDataset):
    r"""Shared helpers for the Kang-Schafer simulations.

    The latent covariates satisfy

    .. math::

        X_1, X_2, X_3, X_4 \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, 1).

    In the correctly specified setting the observed covariates are
    :math:`(X_1, X_2, X_3, X_4)`. In the misspecified setting the dataset exposes
    the nonlinear transformations

    .. math::

        Z_1 = \exp(X_1 / 2),
        \quad
        Z_2 = X_2 / (1 + \exp(X_1)) + 10,

    .. math::

        Z_3 = ((X_1 X_3) / 25 + 0.6)^3,
        \quad
        Z_4 = (X_2 + X_4 + 20)^2.

    The treatment and outcome mechanisms always depend on the latent
    :math:`X` variables, even when only :math:`Z` is observed.
    """

    def __init__(
        self,
        n: int = 1000,
        random_state: int = 42,
        observed_covariates: Literal["correct", "misspecified"] = "correct",
    ):
        if observed_covariates not in {"correct", "misspecified"}:
            raise ValueError("observed_covariates must be 'correct' or 'misspecified'.")

        self.observed_covariates = observed_covariates

        super().__init__(n=n, random_state=random_state)
        self._latent_covariates = None
        self._observed_covariate_array = None
        self._prepare(self.n)

    @property
    def covariate_columns(self) -> list[str]:
        if self.observed_covariates == "misspecified":
            return _TRANSFORMED_COLUMNS
        return _LATENT_COLUMNS

    def _sample_latent_covariates(self, n: int) -> np.ndarray:
        return self._rng.normal(size=(n, len(_LATENT_COLUMNS)))

    def _get_covariates(self) -> pl.DataFrame:
        latent_covariates = self._sample_latent_covariates(self.n)
        return self._set_covariates_from_latent(latent_covariates)

    def _set_covariates_from_latent(
        self, latent_covariates: np.ndarray
    ) -> pl.DataFrame:
        self._latent_covariates = np.asarray(latent_covariates, dtype=float)

        if self.observed_covariates == "misspecified":
            observed_covariates = _transform_covariates(self._latent_covariates)
        else:
            observed_covariates = _latent_to_frame(self._latent_covariates)

        self._observed_covariate_array = observed_covariates.to_numpy()
        return observed_covariates

    def _resolve_latent_covariates(self, covariates) -> np.ndarray:
        covariate_array, columns = _to_numpy_and_columns(
            covariates,
            expected_width=len(_LATENT_COLUMNS),
            name=type(self).__name__,
        )

        if columns == _LATENT_COLUMNS:
            return covariate_array

        if self.observed_covariates == "correct":
            return covariate_array

        if columns == _TRANSFORMED_COLUMNS and self._matches_loaded_covariates(
            covariate_array
        ):
            return self._latent_covariates

        if self._matches_loaded_latent_covariates(covariate_array):
            return covariate_array

        if self._matches_loaded_covariates(covariate_array):
            return self._latent_covariates

        raise ValueError(
            f"{type(self).__name__} can recover the latent covariates only for the "
            "observed rows generated by this dataset instance or for explicitly "
            "provided latent x1-x4 covariates."
        )

    def _matches_loaded_covariates(self, covariate_array: np.ndarray) -> bool:
        return (
            self._observed_covariate_array is not None
            and covariate_array.shape == self._observed_covariate_array.shape
            and np.allclose(covariate_array, self._observed_covariate_array)
        )

    def _matches_loaded_latent_covariates(self, covariate_array: np.ndarray) -> bool:
        return (
            self._latent_covariates is not None
            and covariate_array.shape == self._latent_covariates.shape
            and np.allclose(covariate_array, self._latent_covariates)
        )

    def _prepare(self, n: int = None):
        if n is not None:
            self.n = n

        covariates = self._get_covariates()
        treatments = self._get_treatments(self._latent_covariates)
        outcomes = self._get_outcomes(self._latent_covariates, treatments)

        self._covariates = covariates
        self._treatments = self._to_polars(treatments)
        self._outcomes = outcomes

        return self

    def predict_y(self, covariates, treatments) -> np.ndarray:
        latent_covariates = self._resolve_latent_covariates(covariates)
        treatments = self._check_and_transform_t(treatments)

        n_samples = self._get_n_samples(treatments)
        if latent_covariates.shape[0] != n_samples:
            raise ValueError(
                "predict_y requires covariates and treatments to have the same "
                "number of rows."
            )

        predictions = self._predict_y(latent_covariates, treatments)
        return self._coerce_individual_predictions(predictions, n_samples=n_samples)

    def _baseline(self, latent_covariates: np.ndarray) -> np.ndarray:
        return 210.0 + latent_covariates @ _OUTCOME_WEIGHTS


class KangSchaferBinary(_BaseKangSchafer):
    r"""Binary-treatment Kang-Schafer simulation.

    The treatment follows

    .. math::

        A \mid X \sim \operatorname{Bernoulli}(p(X)),
        \qquad
        p(X) = \operatorname{logit}^{-1}(X_1 - 0.5X_2 + 0.25X_3 + 0.1X_4).

    The outcome model is

    .. math::

        Y \mid A, X \sim \mathcal{N}(210 + A + 27.4X_1 + 13.7X_2 + 13.7X_3 + 13.7X_4, 1).

    The true average treatment effect is exactly 1.
    """

    column_types = {"a": "categorical"}
    TRUE_EFFECT = 1.0

    def _get_treatments(self, covariates) -> pl.DataFrame:
        latent_covariates = self._resolve_latent_covariates(covariates)
        propensity = _sigmoid(latent_covariates @ _PROPENSITY_WEIGHTS)
        treatments = self._rng.binomial(1, propensity).astype(bool)
        return self._to_polars(pl.DataFrame({"a": treatments}))

    def _predict_y(self, covariates, treatments) -> np.ndarray:
        latent_covariates = self._resolve_latent_covariates(covariates)
        treatment_array, _ = _to_numpy_and_columns(
            treatments,
            expected_width=1,
            name=type(self).__name__,
        )
        return self._baseline(latent_covariates) + treatment_array[:, 0]

    def _get_outcomes(self, covariates, treatments) -> pl.DataFrame:
        mean = self._predict_y(covariates, treatments)
        observed = self._rng.normal(loc=mean, scale=1.0)
        return pl.DataFrame({"y": observed.reshape(-1)})

    def get_grid(self) -> pl.DataFrame:
        return self._coerce_treatment_frame(pl.DataFrame({"a": [False, True]}))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"n": 128, "random_state": 7}]


class KangSchaferBinaryMisspecified(KangSchaferBinary):
    """Misspecified observed-covariate version of :class:`KangSchaferBinary`."""

    def __init__(self, n: int = 1000, random_state: int = 42):
        super().__init__(
            n=n,
            random_state=random_state,
            observed_covariates="misspecified",
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"n": 128, "random_state": 7}]


class KangSchaferContinuous(_BaseKangSchafer):
    r"""Continuous-treatment Kang-Schafer simulation.

    The treatment is generated as

    .. math::

        A = X_1 - 0.5X_2 + 0.25X_3 + 0.1X_4 + \varepsilon_A,
        \qquad
        \varepsilon_A \sim \mathcal{N}(0, 1).

    The outcome model is

    .. math::

        Y \mid A, X \sim
        \mathcal{N}(210 + \operatorname{logit}^{-1}(A)
        + 27.4X_1 + 13.7X_2 + 13.7X_3 + 13.7X_4, 1).
    """

    column_types = {"a": "continuous"}

    def _get_treatments(self, covariates) -> pl.DataFrame:
        latent_covariates = self._resolve_latent_covariates(covariates)
        treatment_mean = latent_covariates @ _PROPENSITY_WEIGHTS
        treatments = treatment_mean + self._rng.normal(size=treatment_mean.shape[0])
        return self._to_polars(pl.DataFrame({"a": treatments}))

    def _predict_y(self, covariates, treatments) -> np.ndarray:
        latent_covariates = self._resolve_latent_covariates(covariates)
        treatment_array, _ = _to_numpy_and_columns(
            treatments,
            expected_width=1,
            name=type(self).__name__,
        )
        return self._baseline(latent_covariates) + _sigmoid(treatment_array[:, 0])

    def _get_outcomes(self, covariates, treatments) -> pl.DataFrame:
        mean = self._predict_y(covariates, treatments)
        observed = self._rng.normal(loc=mean, scale=1.0)
        return pl.DataFrame({"y": observed.reshape(-1)})

    def get_grid(self, n: int = 100) -> pl.DataFrame:
        if self._treatments is not None:
            treatment_values = self._treatments.get_column("a").to_numpy()
            lower = float(np.quantile(treatment_values, 0.01))
            upper = float(np.quantile(treatment_values, 0.99))
        else:
            lower, upper = -4.0, 4.0

        return self._coerce_treatment_frame(
            pl.DataFrame({"a": np.linspace(lower, upper, n)})
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"n": 128, "random_state": 7}]


class KangSchaferContinuousMisspecified(KangSchaferContinuous):
    """Misspecified observed-covariate version of :class:`KangSchaferContinuous`."""

    def __init__(self, n: int = 1000, random_state: int = 42):
        super().__init__(
            n=n,
            random_state=random_state,
            observed_covariates="misspecified",
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"n": 128, "random_state": 7}]
