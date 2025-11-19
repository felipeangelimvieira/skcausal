"""Module for direct regression methods.

Direct regression methods try to approximate $\mathbb{E}[Y|X, T]$ directly, using
both X and T as inputs to the regressor. This regression can be weighted, so that
each sample weight is proportional to the inverse of the propensity score $P(t_i|x_i)$.
"""

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
import polars as pl
from skcausal.causal_estimators.base import BaseCausalResponseEstimator
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor

__all__ = ["WeightedDirectRegressor", "WeightedIndividualDirectRegressor"]


class WeightedDirectRegressor(BaseCausalResponseEstimator):
    """
    Perform direct regression with optional weighted samples.

    This method tries to estimate Y(t) directly by fitting Y | X, T using both X and T
    as inputs to the regressor.

    Parameters
    ----------
    outcome_regressor : RegressorMixin
        Sklearn-like regressor to use for estimating the outcome.

    sample_weight_regressor : BaseSampleWeightRegressor, optional
        SampleWeight regressor to use for estimating the sample weights.
        Default is None, which means no sample weights are used.
    """

    _tags = {
        "capability:predicts_individual": False,
        "t_inner_mtype": pl.DataFrame,
    }

    def __init__(
        self,
        outcome_regressor: RegressorMixin,
        sample_weight_regressor: BaseBalancingWeightRegressor = None,
    ):

        self.outcome_regressor = outcome_regressor
        self.sample_weight_regressor = sample_weight_regressor
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray):
        """Fit the outcome model to the data.

        The treatment vector is concatenated to X before passing the inputs to
        outcome regressor

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target variable.
        t : np.ndarray
            Treatment variable.

        Returns
        -------
        self
            The object itself
        """
        if self.sample_weight_regressor is not None:
            self.sample_weight_regressor.fit(X, t)
            self.weights_ = self.sample_weight_regressor.predict_sample_weight(
                X=X, t=t
            ).flatten()
        else:
            self.weights_ = None

        self.fit_kwargs_ = self._prepare_fit_kwargs(X, y, t, self.weights_)
        self.outcome_regressor.fit(**self.fit_kwargs_)
        return self

    def _prepare_fit_kwargs(
        self, X: np.ndarray, y: np.ndarray, t: np.ndarray, weights: np.ndarray
    ) -> tuple:

        dataset = {"X": self._prepare_input_array(X, t), "y": y}

        if weights is not None:
            kwarg_name = "sample_weight"
            if isinstance(self.outcome_regressor, Pipeline):
                kwarg_name = self.outcome_regressor.steps[-1][0] + "__sample_weight"
            dataset[kwarg_name] = weights
        return dataset

    def _prepare_input_array(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Handles how to use X and t as input.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        t : np.ndarray
            Treatment variable.

        Returns
        -------
        np.ndarray
            Array to be passed to outcome regressor
        """
        return self._concat(X, t)

    def _concat(self, X, t):
        if not isinstance(t, np.ndarray):
            t = t.to_numpy()
        return np.concatenate([X, t.reshape((X.shape[0], -1))], axis=1)

    def _predict_individual(
        self, X: np.ndarray, t: np.ndarray
    ) -> np.ndarray:  # -> Any:
        """Predict the individual treatment effect for each sample in X, given t.

        Parameters
        ----------
        X : np.ndarray
            Input data
        t : np.ndarray
            Treatment values

        Returns
        -------
        np.ndarray
            Array with predicted individual treatment effects.
        """
        return self.outcome_regressor.predict(self._prepare_input_array(X, t))

    def _predict_average_treatment_effect(self, X: np.ndarray, t: np.ndarray):
        """Predict the average treatment effect for the given treatment values t.

        Uses `predict_individual`  and takes the average of the sample.

        Parameters
        ----------
        X : np.ndarray
            Input data
        t : np.ndarray
            Treatment values
        """

        ate = self.predict_individual(X, t).mean()
        return ate

    def _predict_adrf(self, X: pd.DataFrame, t: list[float]) -> list[float]:
        """Predict the Average Dose-Response Curve for a list of treatment values.

        Parameters
        ----------
        X : pd.DataFrame
            Input data
        t : list[float]
            List of treatment values

        Returns
        -------
        list[float]
            List of predicted average treatment effects for each treatment value.
        """
        ys = []

        t = t.to_numpy()
        for i in range(t.shape[0]):
            _t = t[i]
            ate = self.outcome_regressor.predict(
                self._prepare_input_array(X, _t * np.ones((X.shape[0], 1)))
            )[0]
            ys.append(ate)
        return ys
