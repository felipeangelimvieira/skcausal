import copy

import numpy as np
import pandas as pd
import polars as pl

from skcausal.density.base import BaseDensityEstimator


__all__ = ["SkproDensityEstimator"]


class SkproDensityEstimator(BaseDensityEstimator):
    """
    Density estimator adapter for skpro probabilistic regressors.

    Parameters
    ----------
    estimator : object
        A fitted-compatible skpro probabilistic regressor implementing
        ``fit(X, y)`` and ``predict_proba(X)``.
    """

    _tags = {
        "supported_t_dtypes": [pl.Float32, pl.Float64],
        "density_kind": "conditional",
        "soft_dependencies": ["skpro"],
    }

    def __init__(self, estimator):
        self.estimator = estimator

        super().__init__()

    def _fit(self, X: pl.DataFrame, t: pl.DataFrame):
        self.estimator_ = copy.deepcopy(self.estimator)
        self.estimator_.fit(self._to_pandas(X), self._to_pandas(t))
        return self

    def _predict_density(self, X: pl.DataFrame, t: pl.DataFrame) -> np.ndarray:
        predictive_distribution = self.estimator_.predict_proba(self._to_pandas(X))
        density = predictive_distribution.pdf(self._to_pandas(t))
        return self._coerce_density_output(density)

    @staticmethod
    def _to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(frame.to_dict(as_series=False))

    @staticmethod
    def _coerce_density_output(density) -> np.ndarray:
        if isinstance(density, pd.DataFrame):
            density_array = density.to_numpy()
        elif isinstance(density, pd.Series):
            density_array = density.to_numpy().reshape(-1, 1)
        else:
            density_array = np.asarray(density)

        if density_array.ndim == 1:
            density_array = density_array.reshape(-1, 1)

        if density_array.ndim != 2:
            raise ValueError(
                "Expected density output to be 1D or 2D, but received array with "
                f"shape {density_array.shape}."
            )

        return density_array.astype(float, copy=False)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from skpro.regression.residual import ResidualDouble

        return [
            {
                "estimator": ResidualDouble(
                    estimator=LinearRegression(),
                    estimator_resid=RandomForestRegressor(
                        n_estimators=10, random_state=0
                    ),
                )
            }
        ]
