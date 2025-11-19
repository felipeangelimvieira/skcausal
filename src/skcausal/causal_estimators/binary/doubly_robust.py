from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator

from skcausal.causal_estimators.base import BaseCausalResponseEstimator
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor

__all__ = [
    "BinaryDoublyRobust",
]


class BinaryDoublyRobust(BaseCausalResponseEstimator):
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

        # training-time T for DR residuals

        is_t1 = self._t.to_numpy().astype(bool).ravel()

        # weights predicted for hypothetical T=1 and T=0 at *X*:
        t1 = pl.DataFrame([True] * X.shape[0], schema=t.schema)
        t0 = pl.DataFrame([False] * X.shape[0], schema=t.schema)

        w1 = np.asarray(self.treatment_regressor_.predict_sample_weight(X, t1)).ravel()
        w0 = np.asarray(self.treatment_regressor_.predict_sample_weight(X, t0)).ravel()

        eps = 1e-8

        # If predict_sample_weight ≈ 1/π(t|x), convert to probabilities:
        p1_raw = np.clip(w1, eps, None) ** (-1)  # π(1|x)
        p0_raw = np.clip(w0, eps, None) ** (-1)  # π(0|x)

        denom = np.clip(p0_raw + p1_raw, eps, None)
        e = p1_raw / denom  # e(x) = P(T=1|X)
        one_minus_e = p0_raw / denom  # P(T=0|X)

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
