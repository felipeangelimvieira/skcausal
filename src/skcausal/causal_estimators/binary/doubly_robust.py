from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator

from skcausal.causal_estimators._density_utils import (
    binary_conditional_probabilities,
    predict_density_array,
)
from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.density.base import BaseDensityEstimator

__all__ = [
    "BinaryDoublyRobust",
]


class BinaryDoublyRobust(BaseAverageCausalResponseEstimator):
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
        outcome_regressor: BaseEstimator,
    ):
        self.treatment_regressor = treatment_regressor
        self.outcome_regressor = outcome_regressor

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

        _t = t.to_numpy().astype(bool).ravel()

        # μ1: fit on treated; μ0: fit on controls
        self.outcome_regressor1_ = deepcopy(self.outcome_regressor)
        self.outcome_regressor1_.fit(X[_t == True], y[_t == True])

        self.outcome_regressor0_ = deepcopy(self.outcome_regressor)
        self.outcome_regressor0_.fit(X[_t == False], y[_t == False])

        # π: clone + fit once
        self.treatment_regressor_ = self.treatment_regressor.clone()
        self.treatment_regressor_.fit(X, t)

        return self

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

        # training-time T for DR residuals

        is_t1 = self._t.to_numpy().astype(bool).ravel()

        # weights predicted for hypothetical T=1 and T=0 at *X*:
        t1 = pl.DataFrame([True] * X.shape[0], schema=t.schema)
        t0 = pl.DataFrame([False] * X.shape[0], schema=t.schema)

        d1 = predict_density_array(self.treatment_regressor_, X, t1).reshape(-1)
        d0 = predict_density_array(self.treatment_regressor_, X, t0).reshape(-1)

        eps = 1e-8

        prior1 = float(is_t1.mean())
        prior0 = 1.0 - prior1
        p0_raw, p1_raw = binary_conditional_probabilities(
            self.treatment_regressor_,
            d0,
            d1,
            marginal_false=prior0,
            marginal_true=prior1,
            eps=eps,
        )
        e = p1_raw  # e(x) = P(T=1|X)
        one_minus_e = p0_raw  # P(T=0|X)

        # outcome models at X
        m1 = np.asarray(self.outcome_regressor1_.predict(X)).ravel()  # μ1(x)
        m0 = np.asarray(self.outcome_regressor0_.predict(X)).ravel()  # μ0(x)
        y = np.asarray(self._y).ravel()

        # DR functionals:
        y1 = np.mean(m1 + is_t1 * (y - m1) / np.clip(e, eps, None))
        y0 = np.mean(m0 + (1 - is_t1) * (y - m0) / np.clip(one_minus_e, eps, None))

        # map requested t values to averages
        out = []
        for tval in t.to_numpy().ravel():
            out.append(y1 if bool(tval) else y0)
        return out
