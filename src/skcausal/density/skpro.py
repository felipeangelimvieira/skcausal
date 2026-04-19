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
        "backend": "pandas",
        "capability:t_type": ["continuous"],
        "density_kind": "conditional",
        "soft_dependencies": ["skpro"],
    }

    def __init__(self, estimator):
        self.estimator = estimator

        super().__init__()

    def _fit(self, X: pl.DataFrame, t: pl.DataFrame):
        self.estimator_ = copy.deepcopy(self.estimator)
        self.estimator_.fit(X, t)
        return self

    def _predict_density(self, X: pl.DataFrame, t: pl.DataFrame) -> np.ndarray:
        predictive_distribution = self.estimator_.predict_proba(X)
        density = predictive_distribution.pdf(t)
        return density

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
