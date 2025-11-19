from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.kernel_ridge import KernelRidge

from skcausal.causal_estimators.base import BaseCausalResponseEstimator, to_dummies
from skcausal.utils.polars import convert_categorical_to_dummies
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor
from sklearn.neighbors import KernelDensity


__all__ = [
    "BinaryPropensityWeighting",
]


class BinaryPropensityWeighting(BaseCausalResponseEstimator):
    """
    Uses Propensity Score Weighting to forecast the average treatment effect of Discrete Treatments.

    Parameters
    ----------
    treatment_regressor : BaseSampleWeightRegressor
        Regressor to estimate the propensity score.
    """

    _tags = {
        "capability:predicts_individual": False,
        "capability:supports_multidimensional_treatment": False,
        "t_inner_mtype": pl.DataFrame,
        "store_X": True,
        "one_hot_encode_enum_columns": False,
    }

    def __init__(
        self,
        treatment_regressor: BaseBalancingWeightRegressor,
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

        return np.array(self._predict_adrf(X, t)).reshape((-1, 1)).mean()

    def _predict_adrf(self, X: np.ndarray, t: pl.DataFrame) -> list[float]:
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
        w1 = self.treatment_regressor_.predict_sample_weight(X, t1)

        t0 = pl.DataFrame([False] * X.shape[0], schema=t.schema)
        w0 = self.treatment_regressor_.predict_sample_weight(X, t0)

        is_stabilized = (
            self.treatment_regressor_.get_tag("balancing_weight_type") == "stabilized"
        )

        p1 = w1 ** (-1)
        p0 = w0 ** (-1)

        prior1 = (_t == True).mean()
        prior0 = (_t == False).mean()

        if is_stabilized:
            prior1 = 1.0
            prior0 = 1.0
        else:
            print(p1[:2], p0[:2])
            p1_raw = p1
            p0_raw = p0
            denom = p0_raw + p1_raw
            p1, p0 = p1_raw / denom, p0_raw / denom
            print(p1[:2], p0[:2])

        y1 = (self._y[_t == True] / p1[_t == True]).mean() * prior1
        y0 = (self._y[_t == False] / p0[_t == False]).mean() * prior0
        out = []
        for tval in t.to_numpy().flatten():
            if tval == True:
                out.append(y1)
            else:
                out.append(y0)
        return out
