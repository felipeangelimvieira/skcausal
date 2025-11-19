import datetime
from copy import deepcopy

import warnings
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator

from skcausal.causal_estimators.base import BaseCausalResponseEstimator, to_dummies
from skcausal.utils.polars import convert_categorical_to_dummies
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor


class GPS(BaseCausalResponseEstimator):
    """
    The Generalized Propensity Score (GPS) method of Hirano and Imbens (2004).

    Uses the Propensity Score to create an estimation Y | P(T|X), T, then
    averages the results over the samples to obtain the expected value of Y.


    Parameters
    ----------
    treatment_regressor : BaseSampleWeightRegressor
        Regressor to estimate the propensity score.

    outcome_regressor : BaseEstimator
        Regressor to estimate the outcome

    propensity_split_ratio : float, default=1.0
        Fraction of samples reserved to fit the treatment regressor.

    include_in_outcome_dataset : {"test", "both"}, default="both"
        Which samples are used to fit the outcome regressor; either only the
        held-out (test) samples or both train and test samples.
    """

    _tags = {
        "capability:predicts_individual": True,
        "capability:supports_multidimensional_treatment": True,
        "t_inner_mtype": pl.DataFrame,
        "store_X": True,
        "one_hot_encode_enum_columns": False,
    }

    def __init__(
        self,
        treatment_regressor: BaseBalancingWeightRegressor,
        outcome_regressor: BaseEstimator,
        invert_weight: bool = True,
        propensity_split_ratio: float = 0.8,
        include_in_outcome_dataset: str = "both",
        random_state=0,
    ):
        self.treatment_regressor = treatment_regressor
        self.outcome_regressor = outcome_regressor
        self.invert_weight = invert_weight
        self.propensity_split_ratio = propensity_split_ratio
        self.include_in_outcome_dataset = include_in_outcome_dataset
        self.random_state = random_state

        super().__init__()

    def _set_random_state_params(self, estimator, random_state):
        """Set random_state on all nested estimators that expose the param."""

        if estimator is None or not hasattr(estimator, "get_params"):
            return estimator

        params = estimator.get_params(deep=True)
        random_state_updates = {
            name: random_state for name in params if name.endswith("random_state")
        }
        if random_state_updates:
            estimator.set_params(**random_state_updates)

        return estimator

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

        include_option = str(self.include_in_outcome_dataset).lower()
        if include_option not in {"test", "both"}:
            raise ValueError(
                "include_in_outcome_dataset must be either 'test' or 'both'."
            )

        ratio = float(self.propensity_split_ratio)
        if not 0 < ratio <= 1:
            raise ValueError("propensity_split_ratio must be in the interval (0, 1].")

        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("GPS estimator requires at least one sample to fit.")

        treatment_sample_count = int(np.floor(ratio * n_samples))
        if treatment_sample_count <= 0:
            treatment_sample_count = 1
        treatment_sample_count = min(treatment_sample_count, n_samples)

        rng = np.random.RandomState(self.random_state)
        if treatment_sample_count >= n_samples:
            treatment_indices = np.arange(n_samples)
        else:
            treatment_indices = np.sort(
                rng.choice(n_samples, size=treatment_sample_count, replace=False)
            )

        holdout_mask = np.ones(n_samples, dtype=bool)
        holdout_mask[treatment_indices] = False
        holdout_indices = np.where(holdout_mask)[0]

        if include_option == "test":
            if holdout_indices.size == 0:
                raise ValueError(
                    "include_in_outcome_dataset='test' requires at least one hold-out sample."
                )
            outcome_indices = holdout_indices
        else:
            outcome_indices = np.arange(n_samples)

        self.include_in_outcome_dataset_ = include_option
        self.propensity_split_ratio_ = ratio
        self._propensity_train_indices = treatment_indices
        self._propensity_holdout_indices = holdout_indices

        if treatment_indices.size == n_samples:
            treatment_X = X
            treatment_t = t
        else:
            treatment_X = X[treatment_indices]
            treatment_t = t[treatment_indices.tolist()]

        # Split samples between propensity (treatment) and outcome stages
        self.outcome_regressor_ = deepcopy(self.outcome_regressor)
        self._set_random_state_params(self.outcome_regressor_, self.random_state)

        self.treatment_regressor_ = self.treatment_regressor
        if self.treatment_regressor is not None:
            self.treatment_regressor_ = self.treatment_regressor.clone()
            self._set_random_state_params(self.treatment_regressor_, self.random_state)
            self.treatment_regressor_.fit(treatment_X, treatment_t)

        if outcome_indices.size == n_samples:
            outcome_X = X
            outcome_t = t
            outcome_y = y
        else:
            outcome_X = X[outcome_indices]
            outcome_t = t[outcome_indices.tolist()]
            outcome_y = y[outcome_indices]

        treat_gps = self.make_treatment_gps_array(outcome_X, outcome_t)

        self.outcome_regressor_.fit(treat_gps, outcome_y)

    def make_treatment_gps_array(self, X: np.ndarray, t: pl.DataFrame) -> np.ndarray:
        """Return the pair (GPS, T) for each sample in X.

        Parameters
        ----------
        X : np.ndarray
            The input data
        t : np.ndarray
            The treatment values

        Returns
        -------
        np.ndarray
            An array of shape (n_samples, t.shape[1]+1) containing the GPS and T values for each sample.
        """

        # Sample weight is 1/P(T|X), so we need to invert it
        if self.treatment_regressor is not None:
            sample_weight = self.treatment_regressor_.predict_sample_weight(X, t)
            if sample_weight.std() < 1e-9:
                print(
                    "The propensity score is constant. This may be due to the fact that the treatment regressor is not able to predict the treatment values."
                )
                warnings.warn(
                    "The propensity score is constant. This may be due to the fact that the treatment regressor is not able to predict the treatment values."
                )
                raise ValueError(
                    "The propensity score is constant. This may be due to the fact that the treatment regressor is not able to predict the treatment values."
                )
        else:
            sample_weight = np.ones(X.shape[0])

        if self.invert_weight:
            gps = (sample_weight + 1e-8) ** -1
        else:
            gps = sample_weight
        if not isinstance(gps, np.ndarray):
            gps = gps.to_numpy()

        if self.get_tag("one_hot_encode_enum_columns", False):
            for col, dtype in zip(t.columns, t.dtypes):
                if not dtype.is_numeric():
                    t = to_dummies(t, col)

        t = convert_categorical_to_dummies(t)

        t = t.to_numpy().astype(np.float32)

        # Generate random normal like t
        # t = np.random.normal(size=(X.shape[0], t.shape[1]))

        treat_gps = np.concatenate(
            (
                gps.reshape((-1, 1)),
                t,
            ),
            axis=1,
        )
        return treat_gps

    def _predict_individual(self, X: np.ndarray, t: np.ndarray):
        """Predict individual treatment effect

        Parameters
        ----------
        X : np.ndarray
            The input data
        t : np.ndarray
            The treatment values

        Returns
        -------
        np.ndarray
            The predicted individual treatment effect for each sample in X.
        """

        treat_gps = self.make_treatment_gps_array(X, t)
        return self.outcome_regressor_.predict(treat_gps)

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
            The treatment values

        Returns
        -------
        list[float]
            The average response for each treatment value in t.
        """

        effects = []
        repeated_treat_values = t[np.repeat(np.arange(t.shape[0]), X.shape[0])]
        repeated_X = np.tile(X, (len(t), 1))
        treat_gps = self.make_treatment_gps_array(
            repeated_X, repeated_treat_values
        ).reshape((len(t), X.shape[0], -1))
        for i in range(t.shape[0]):

            effect = self.outcome_regressor_.predict(treat_gps[i]).mean()
            effects.append(effect)

        return effects
