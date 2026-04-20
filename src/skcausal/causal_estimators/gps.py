from typing import Optional
from copy import deepcopy

import warnings
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator

from skcausal.causal_estimators.base import (
    BaseAverageCausalResponseEstimator,
)
from skcausal.causal_estimators._density_utils import predict_density_array
from skcausal.density.base import BaseDensityEstimator
from skcausal.utils.polars import convert_categorical_to_dummies, to_dummies
from sklearn.model_selection import KFold


class GPS(BaseAverageCausalResponseEstimator):
    """
    The Generalized Propensity Score (GPS) method of Hirano and Imbens (2004).

    Uses the Propensity Score to create an estimation Y | P(T|X), T, then
    averages the results over the samples to obtain the expected value of Y.

    The estimation of P(T|X) is executed using cross-validation, and we store
    out-of-sample estimates to later feed Y | P(T|X), T for training.

    When predicting, we use a density regressor fitted on the whole dataset to
    estimate P(T|X) for the new samples, and then feed those estimates to the
    outcome regressor to predict Y.


    Parameters
    ----------
    density_regressor : BaseDensityEstimator
        Density estimator used to estimate the propensity score.

    outcome_regressor : BaseEstimator
        Regressor to estimate the outcome

    cv : int, default=5
        Number of cross-validation folds to use when estimating the
        propensity score.
    """

    _tags = {
        "capability:multidimensional_treatment": True,
        "backend": "polars",
        "capability:t_type": ["continuous", "categorical"],
        "one_hot_encode_enum_columns": False,
    }

    def __init__(
        self,
        density_regressor: BaseDensityEstimator,
        outcome_regressor: BaseEstimator,
        cv: int = 5,
        random_state=0,
    ):
        if density_regressor is None:
            raise ValueError("GPSOut requires a non-null density_regressor.")

        self.density_regressor = density_regressor
        self.outcome_regressor = outcome_regressor
        self.cv = cv
        self.random_state = random_state

        super().__init__()

    def make_treatment_gps_array(
        self,
        X: pl.DataFrame,
        t: pl.DataFrame,
        density_regressor: Optional[BaseDensityEstimator] = None,
    ) -> np.ndarray:
        """Return the pair (GPS, T) for each sample in X."""

        density_regressor = (
            self.treatment_regressor_
            if density_regressor is None
            else density_regressor
        )

        if density_regressor is not None:
            gps = predict_density_array(density_regressor, X, t)
            if gps.std() < 1e-9:
                message = (
                    "The estimated treatment density is constant. "
                    "The density estimator may not be learning the treatment distribution."
                )
                warnings.warn(message)
                raise ValueError(message)
        else:
            gps = np.ones((X.height, 1), dtype=float)

        if self.get_tag("one_hot_encode_enum_columns", False):
            for col, dtype in zip(t.columns, t.dtypes):
                if dtype == pl.Enum or dtype == pl.Categorical:
                    t = to_dummies(t, col)

        t = convert_categorical_to_dummies(t)
        t = t.to_numpy().astype(np.float32)

        return np.concatenate((gps.reshape((-1, 1)), t), axis=1)

    def _fit(self, X: pl.DataFrame, t: pl.DataFrame, y: pl.DataFrame):
        """Fit the outcome model on out-of-fold GPS features."""

        self._X = X
        self._y = y
        self._t = t

        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("GPSOut requires at least one sample to fit.")

        if isinstance(self.cv, bool) or not isinstance(self.cv, (int, np.integer)):
            raise TypeError("cv must be an integer greater than or equal to 2.")

        n_splits = int(self.cv)
        if n_splits < 2:
            raise ValueError("cv must be an integer greater than or equal to 2.")

        self.outcome_regressor_ = deepcopy(self.outcome_regressor)

        self.fold_density_regressors_ = []
        self.fold_test_indices_ = []
        self.oof_test_indices_ = []
        self.cv_splitter_ = None
        oof_treatment_gps = []

        if n_splits > n_samples:
            raise ValueError(
                "cv cannot be greater than the number of training samples when density_regressor is provided."
            )

        self.cv_splitter_ = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for train_indices, test_indices in self.cv_splitter_.split(X):
            fold_density_regressor = self.density_regressor.clone()

            train_t = t[train_indices.tolist()]
            test_t = t[test_indices.tolist()]

            fold_density_regressor.fit(X[train_indices], train_t)
            self.fold_density_regressors_.append(fold_density_regressor)
            self.fold_test_indices_.append(np.asarray(test_indices, dtype=int))
            self.oof_test_indices_.extend(test_indices.tolist())

            fold_treatment_gps = self.make_treatment_gps_array(
                X[test_indices],
                test_t,
                density_regressor=fold_density_regressor,
            )
            oof_treatment_gps.append(fold_treatment_gps)

        self.density_regressor_ = self.density_regressor.clone()
        self.density_regressor_.fit(X, t)
        self.treatment_regressor_ = self.density_regressor_

        self.oof_test_indices_ = np.asarray(self.oof_test_indices_, dtype=int)
        self.oof_treatment_gps_ = np.concatenate(oof_treatment_gps, axis=0)
        outcome_y = y[self.oof_test_indices_]

        self.outcome_regressor_.fit(self.oof_treatment_gps_, outcome_y)

        return self

    def _predict(self, X: pl.DataFrame, t: pl.DataFrame) -> list[float]:
        """Predict the average response for each treatment value in t."""

        effects = []
        n_samples = X.height
        repeated_treat_values = t[np.repeat(np.arange(t.shape[0]), n_samples)]
        repeated_X = pl.concat([X] * len(t), how="vertical")
        treat_gps = self.make_treatment_gps_array(
            repeated_X,
            repeated_treat_values,
        ).reshape((len(t), n_samples, -1))

        for i in range(t.shape[0]):
            effect = self.outcome_regressor_.predict(treat_gps[i]).mean()
            effects.append(effect)

        return effects

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from skcausal.density.naive import NaiveDensityEstimator
        from sklearn.linear_model import LinearRegression

        return [
            {
                "density_regressor": NaiveDensityEstimator(),
                "outcome_regressor": LinearRegression(),
                "cv": 2,
            }
        ]
