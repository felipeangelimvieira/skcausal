from typing import Literal

import numpy as np
import polars as pl

from skcausal.datasets.base import BaseSyntheticDataset

__all__ = ["PolynomialDataset"]


def _expand_polynomial(values: np.ndarray, degree: int) -> np.ndarray:
    if degree < 1:
        raise ValueError("degree must be at least 1")

    blocks = [values]
    for exponent in range(2, degree + 1):
        blocks.append(values**exponent)
    return np.concatenate(blocks, axis=1)


def _as_2d_float_array(values) -> np.ndarray:
    if isinstance(values, pl.DataFrame):
        values = values.to_numpy()

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped_values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped_values))


class PolynomialDataset(BaseSyntheticDataset):
    """
    Polynomial X->T and (X,T)->Y

    Generates a dataset given number of polynomial expansion and number of
    treatments.
    """

    def __init__(
        self,
        n=1000,
        x_to_t_degree=2,
        x_to_y_degree=2,
        t_to_y_degree=2,
        covariate_dim_types=None,
        treatment_dim_types=None,
        seed=42,
        covariate_effect_scale=1.0,
        treatment_effect_scale=1.0,
        joint_effect_scale=1.0,
    ):
        self.x_to_t_degree = x_to_t_degree
        self.x_to_y_degree = x_to_y_degree
        self.t_to_y_degree = t_to_y_degree
        self.covariate_dim_types = covariate_dim_types
        self.treatment_dim_types = treatment_dim_types
        self.covariate_effect_scale = covariate_effect_scale
        self.treatment_effect_scale = treatment_effect_scale
        self.joint_effect_scale = joint_effect_scale

        super().__init__(n=n, seed=seed)

        self._covariate_dim_types = covariate_dim_types
        if covariate_dim_types is None:
            self._covariate_dim_types = ["continuous", "continuous", "binary"]

        self._treatment_dim_types = treatment_dim_types
        if treatment_dim_types is None:
            self._treatment_dim_types = ["continuous", "continuous", "binary"]

        self._beta_X_to_t = self._rng.laplace(
            size=len(self._covariate_dim_types) * self.x_to_t_degree, scale=1
        )
        self._beta_X_to_y = self._rng.normal(
            size=len(self._covariate_dim_types) * self.x_to_y_degree, scale=1
        )
        self._beta_T_to_y = self._rng.normal(
            size=len(self._treatment_dim_types) * self.t_to_y_degree, scale=1
        )
        self._set_treatment_schema()
        self._prepare(self.n)

    def _set_treatment_schema(self):
        if self._treatment_dim_types is None:
            return

        schema = {}
        for dim_id, dtype in enumerate(self._treatment_dim_types):
            if dtype == "binary":
                schema[f"t{dim_id}"] = pl.Boolean
            elif dtype == "continuous":
                schema[f"t{dim_id}"] = pl.Float64
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

        schema = pl.Schema(schema)
        self.TREATMENT_SCHEMA = schema

    def _sample_dtype(self, dtype: Literal["continuous", "binary"], n: int):
        if dtype == "continuous":
            return self._rng.normal(size=n)
        elif dtype == "binary":
            return self._rng.binomial(1, 0.5, size=n)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def generate_columns(
        self, n: int, dim_types: list[Literal["continuous", "binary"]]
    ):

        arr = np.empty((n, len(dim_types)))
        for dim_id, dtype in enumerate(dim_types):
            arr[:, dim_id] = self._sample_dtype(dtype, n=n)
        return arr

    def _get_covariates(self):
        return self.generate_columns(n=self.n, dim_types=self._covariate_dim_types)

    def _covariate_features(self, covariates, *, degree: int) -> np.ndarray:
        return _expand_polynomial(_as_2d_float_array(covariates), degree=degree)

    def _treatment_features(self, treatments) -> np.ndarray:
        return _expand_polynomial(
            _as_2d_float_array(treatments), degree=self.t_to_y_degree
        )

    def _get_treatments(self, covariates):
        arr = self._covariate_features(covariates, degree=self.x_to_t_degree)
        arr = arr @ self._beta_X_to_t

        columns = {}
        for dim_id, dtype in enumerate(self._treatment_dim_types):
            arr_i = arr ** (dim_id + 1)
            if dtype == "binary":
                arr_i = _sigmoid(arr_i)
                columns[f"t{dim_id}"] = self._rng.binomial(
                    1, arr_i, size=arr_i.shape
                ).astype(bool)
            elif dtype == "continuous":
                columns[f"t{dim_id}"] = arr_i
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

        return pl.DataFrame(columns, schema=self.TREATMENT_SCHEMA)

    def _predict_y(self, covariates, treatments):
        covariate_direct_effect = (
            self._covariate_features(covariates, degree=self.x_to_y_degree)
            @ self._beta_X_to_y
        )
        treatment_direct_effect = (
            self._treatment_features(treatments) @ self._beta_T_to_y
        )

        out = (
            covariate_direct_effect * self.covariate_effect_scale
            + treatment_direct_effect * self.treatment_effect_scale
            + covariate_direct_effect
            * treatment_direct_effect
            * self.joint_effect_scale
        )

        return out

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"n": 100}]
