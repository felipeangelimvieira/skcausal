from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.kernel_ridge import KernelRidge

from skcausal.causal_estimators._density_utils import (
    is_stabilized_density,
    predict_density_array,
)
from skcausal.causal_estimators.base import (
    BaseAverageCausalResponseEstimator,
    to_dummies,
)
from skcausal.utils.polars import convert_categorical_to_dummies
from skcausal.density.base import BaseDensityEstimator
from sklearn.neighbors import KernelDensity


__all__ = [
    "BinaryPropensityWeighting",
]


class BinaryPropensityWeighting(BaseAverageCausalResponseEstimator):
    """
    Uses Propensity Score Weighting to forecast the average treatment effect of Discrete Treatments.

    Parameters
    ----------
    treatment_regressor : BaseDensityEstimator
        Density estimator used to estimate treatment propensities.
    """

    _tags = {
        "capability:multidimensional_treatment": False,
        "t_inner_mtype": pl.DataFrame,
        "store_X": True,
        "one_hot_encode_enum_columns": False,
    }

    def __init__(
        self,
        treatment_regressor: BaseDensityEstimator,
    ):
        self.treatment_regressor = treatment_regressor

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray, t: pl.DataFrame):
        """Fits the GPS estimator.


        First, fits the treatment regressor to estimate the propensity score.
        Then, fits the outcome regressor to estimate the outcome.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target variable.
        t : np.ndarray

        Returns
        -------
        self
            The object itself
        """

        self._X = X
        self._y = y
        self._t = t

        self.treatment_regressor_ = self.treatment_regressor
        if self.treatment_regressor is not None:
            self.treatment_regressor_ = self.treatment_regressor.clone()
            self.treatment_regressor_.fit(X, t)

    def _predict_average_treatment_effect(self, X, t) -> float:
        """Predict the average treatment effect for the given treatment values t.

        Parameters
        ----------
        X : np.ndarray
            The input data
        t : np.ndarray
            The treatment values

        Returns
        -------
        float
            The average treatment effect for the given treatment values t.
        """

        return np.array(self._predict(X, t)).reshape((-1, 1)).mean()

    def _predict(self, X: np.ndarray, t: pl.DataFrame) -> list[float]:
        """
        Predict the average response for each treatment value in t.

        Parameters
        ----------
        X : np.ndarray
            The input data
        t : list[float]
            The treatment values. Ignored since for binary treatments the values can only be False and True

        Returns
        -------
        list[float]
            The average response for each treatment value in t.
        """

        #        assert len(t) == 2, "PropensityWeighting only supports binary treatments"

        _t = self._t.to_numpy().astype(bool).flatten()

        t1 = pl.DataFrame([True] * X.shape[0], schema=t.schema)
        p1 = predict_density_array(self.treatment_regressor_, X, t1).reshape(-1)

        t0 = pl.DataFrame([False] * X.shape[0], schema=t.schema)
        p0 = predict_density_array(self.treatment_regressor_, X, t0).reshape(-1)

        is_stabilized = is_stabilized_density(self.treatment_regressor_)

        prior1 = (_t == True).mean()
        prior0 = (_t == False).mean()

        if is_stabilized:
            prior1 = 1.0
            prior0 = 1.0
        else:
            p1_raw = p1
            p0_raw = p0
            denom = np.clip(p0_raw + p1_raw, 1e-8, None)
            p1, p0 = p1_raw / denom, p0_raw / denom

        y1 = (self._y[_t == True] / np.clip(p1[_t == True], 1e-8, None)).mean() * prior1
        y0 = (
            self._y[_t == False] / np.clip(p0[_t == False], 1e-8, None)
        ).mean() * prior0
        out = []
        for tval in t.to_numpy().flatten():
            if tval == True:
                out.append(y1)
            else:
                out.append(y0)
        return out
