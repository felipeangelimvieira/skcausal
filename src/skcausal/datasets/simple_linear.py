from typing import Literal

import numpy as np
import polars as pl

from skcausal.datasets.base import BaseSyntheticDataset

__all__ = ["SimpleLinearDataset"]


def _as_2d_float_array(values) -> np.ndarray:
    if isinstance(values, pl.DataFrame):
        values = values.to_numpy()

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


class SimpleLinearDataset(BaseSyntheticDataset):
    """Simple linear synthetic dataset with configurable treatment types."""

    def __init__(
        self,
        n=1000,
        t_types=("continuous",),
        n_features=5,
        scale=1.0,
        random_state=42,
    ):
        normalized_t_types = self._normalize_t_types(t_types)
        if n_features < 1:
            raise ValueError("n_features must be at least 1")
        if scale <= 0:
            raise ValueError("scale must be positive")

        self.t_types = normalized_t_types
        self.n_features = n_features
        self.scale = scale
        self.random_state = random_state

        super().__init__(n=n, seed=random_state)

        self._t_types = normalized_t_types
        self.beta_x_ = self._rng.normal(size=self.n_features)
        self.beta_t_ = self._rng.normal(size=len(self._t_types))
        self._set_treatment_schema()
        self._prepare(self.n)

    @staticmethod
    def _normalize_t_types(
        t_types,
    ) -> tuple[Literal["continuous", "binary"], ...]:
        if isinstance(t_types, str):
            normalized = (t_types,)
        else:
            normalized = tuple(t_types)

        if not normalized:
            raise ValueError("t_types must contain at least one treatment type")

        supported_types = {"continuous", "binary"}
        invalid_types = [
            t_type for t_type in normalized if t_type not in supported_types
        ]
        if invalid_types:
            raise ValueError(
                "Unsupported treatment types: "
                + ", ".join(invalid_types)
                + ". Supported values are 'continuous' and 'binary'."
            )

        return normalized

    def _set_treatment_schema(self):
        schema = {}
        for dim_id, dtype in enumerate(self._t_types):
            if dtype == "continuous":
                schema[f"t{dim_id}"] = pl.Float64
            else:
                schema[f"t{dim_id}"] = pl.Boolean

        self.TREATMENT_SCHEMA = pl.Schema(schema)

    def _get_covariates(self) -> np.ndarray:
        return self._rng.normal(size=(self.n, self.n_features))

    def _linear_covariate_effect(self, covariates) -> np.ndarray:
        covariates_array = _as_2d_float_array(covariates)
        return covariates_array @ self.beta_x_

    def _get_treatments(self, covariates) -> pl.DataFrame:
        linear_mean = self._linear_covariate_effect(covariates)
        covariate_threshold = _as_2d_float_array(covariates).mean()

        columns = {}
        for dim_id, dtype in enumerate(self._t_types):
            if dtype == "continuous":
                columns[f"t{dim_id}"] = self._rng.normal(
                    loc=linear_mean,
                    scale=self.scale,
                    size=linear_mean.shape,
                )
            else:
                probabilities = (linear_mean > covariate_threshold).astype(int)
                columns[f"t{dim_id}"] = self._rng.binomial(
                    1,
                    probabilities,
                    size=linear_mean.shape,
                ).astype(bool)

        return pl.DataFrame(columns, schema=self.TREATMENT_SCHEMA)

    def _predict_y(self, covariates, treatments) -> np.ndarray:
        covariate_effect = self._linear_covariate_effect(covariates)
        treatment_effect = _as_2d_float_array(treatments) @ self.beta_t_
        return covariate_effect + treatment_effect

    def _inject_outcome_noise(self, expected_outcomes, covariates, treatments):
        return expected_outcomes

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [
            {"n": 100, "t_types": ("continuous",), "n_features": 4},
            {
                "n": 100,
                "t_types": ("continuous", "binary"),
                "n_features": 3,
                "random_state": 7,
            },
        ]
