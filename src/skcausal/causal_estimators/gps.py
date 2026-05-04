from typing import Optional
from copy import deepcopy

import warnings
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator

from skcausal.causal_estimators.base import (
    BaseAverageCausalResponseEstimator,
)
from skcausal.causal_estimators._density_utils import predict_density_array
from skcausal.density.base import BaseDensityEstimator
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
        Regressor to estimate the outcome. When the treatment includes
        categorical columns, any required encoding should be handled by this
        regressor, typically via a preprocessing pipeline.

    cv : int, default=5
        Number of cross-validation folds to use when estimating the
        propensity score.
    max_samples_predict : int, optional
        Maximum number of covariate rows to average over at prediction time.
        If provided and ``X`` contains more rows than this value, GPS samples
        rows without replacement before constructing the score-and-treatment
        features. This can reduce prediction cost for large evaluation samples.
    """

    _tags = {
        "capability:multidimensional_treatment": True,
        "backend": "polars",
        "capability:t_type": ["continuous", "categorical"],
    }

    def __init__(
        self,
        density_regressor: BaseDensityEstimator,
        outcome_regressor: BaseEstimator,
        cv: int = 5,
        max_samples_predict: Optional[int] = None,
        random_state=0,
    ):
        if density_regressor is None:
            raise ValueError("GPSOut requires a non-null density_regressor.")

        self.density_regressor = density_regressor
        self.outcome_regressor = outcome_regressor
        self.cv = cv
        self.max_samples_predict = max_samples_predict
        self.random_state = random_state

        super().__init__()

        self._max_samples_predict = self._coerce_max_samples_predict(
            self.max_samples_predict
        )

    @staticmethod
    def _coerce_max_samples_predict(
        max_samples_predict: Optional[int],
    ) -> Optional[int]:
        if max_samples_predict is None:
            return None

        if isinstance(max_samples_predict, bool) or not isinstance(
            max_samples_predict, (int, np.integer)
        ):
            raise TypeError("max_samples_predict must be an integer or None.")
        if int(max_samples_predict) < 1:
            raise ValueError("max_samples_predict must be greater than or equal to 1.")

        return int(max_samples_predict)

    def _select_prediction_indices(self, n_samples: int) -> np.ndarray:
        if self._max_samples_predict is None or self._max_samples_predict >= n_samples:
            return np.arange(n_samples, dtype=int)

        rng = np.random.default_rng(self.random_state)
        selected = rng.choice(n_samples, size=self._max_samples_predict, replace=False)
        return np.sort(selected.astype(int))

    def make_treatment_gps_array(
        self,
        X: pl.DataFrame,
        t: pl.DataFrame,
        density_regressor: Optional[BaseDensityEstimator] = None,
    ) -> pd.DataFrame:
        """Return outcome-model features with GPS and raw treatment columns.

        Categorical treatment columns are preserved as-is so any encoding is
        delegated to the user-supplied outcome regressor, typically via a
        preprocessing pipeline.
        """

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

        gps_frame = pd.DataFrame({"gps": np.asarray(gps, dtype=float).reshape(-1)})
        treatment_frame = t.to_pandas().reset_index(drop=True)

        return pd.concat([gps_frame, treatment_frame], axis=1)

    def _fit(self, X: pl.DataFrame, t: pl.DataFrame, y: pl.DataFrame):
        """Fit the outcome model on out-of-fold GPS features."""

        self._X = X
        self._y = y
        self._t = t

        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("GPSOut requires at least one sample to fit.")

        if isinstance(self.cv, bool) or not isinstance(self.cv, (int, np.integer)):
            raise TypeError(
                "cv must be an integer equal to 0 or greater than or equal to 2."
            )

        n_splits = int(self.cv)
        if n_splits != 0 and n_splits < 2:
            raise ValueError(
                "cv must be an integer equal to 0 or greater than or equal to 2."
            )

        self.outcome_regressor_ = deepcopy(self.outcome_regressor)

        self.fold_density_regressors_ = []
        self.fold_test_indices_ = []
        self.oof_test_indices_ = []
        self.cv_splitter_ = None
        oof_treatment_gps = []

        if n_splits == 0:
            self.density_regressor_ = self.density_regressor.clone()
            self.density_regressor_.fit(X, t)
            self.treatment_regressor_ = self.density_regressor_

            insample_gps = self.make_treatment_gps_array(X, t)
            self.oof_test_indices_ = np.arange(n_samples, dtype=int)
            self.oof_treatment_gps_ = insample_gps

            self.outcome_regressor_.fit(insample_gps, y)
            return self

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
        self.oof_treatment_gps_ = pd.concat(
            oof_treatment_gps, axis=0, ignore_index=True
        )
        outcome_y = y[self.oof_test_indices_]

        self.outcome_regressor_.fit(self.oof_treatment_gps_, outcome_y)

        return self

    def _predict(self, t: pl.DataFrame) -> list[float]:
        """Predict the average response for each treatment value in t."""

        X = self._get_fit_X()
        prediction_indices = self._select_prediction_indices(X.height)
        X = X[prediction_indices]

        effects = []
        n_samples = X.height
        repeated_treat_values = t[np.repeat(np.arange(t.shape[0]), n_samples)]
        repeated_X = pl.concat([X] * len(t), how="vertical")
        treat_gps = self.make_treatment_gps_array(
            repeated_X,
            repeated_treat_values,
        )

        for i in range(t.shape[0]):
            start = i * n_samples
            stop = start + n_samples
            effect = self.outcome_regressor_.predict(treat_gps.iloc[start:stop]).mean()
            effects.append(effect)

        return effects

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from skcausal.density.naive import NaiveDensityEstimator
        from sklearn.compose import ColumnTransformer, make_column_selector
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "encode_categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                    make_column_selector(
                        dtype_include=["category", "object", "string"]
                    ),
                )
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        return [
            {
                "density_regressor": NaiveDensityEstimator(),
                "outcome_regressor": make_pipeline(preprocessor, LinearRegression()),
                "cv": 2,
            }
        ]
