"""Module for direct regression methods.

Direct regression methods try to approximate $\mathbb{E}[Y|X, T]$ directly, using
both X and T as inputs to the regressor. This regression can be weighted, so that
each sample weight is proportional to the inverse treatment density $1 / P(t_i|x_i)$.
"""

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.causal_estimators._density_utils import predict_inverse_density_weight
from skcausal.density.base import BaseDensityEstimator

__all__ = ["DirectRegressor"]


class DirectRegressor(BaseAverageCausalResponseEstimator):
    """
    Perform direct regression with optional weighted samples.

    This method tries to estimate Y(t) directly by fitting Y | X, T using both X and T
    as inputs to the regressor.

    Parameters
    ----------
    outcome_regressor : RegressorMixin
        Sklearn-like regressor to use for estimating the outcome.

    sample_weight_regressor : BaseDensityEstimator, optional
        Density estimator used to derive inverse-density sample weights.
        Default is None, which means no sample weights are used.
    """

    _tags = {"backend": "pandas"}

    def __init__(
        self,
        outcome_regressor: RegressorMixin,
        sample_weight_regressor: BaseDensityEstimator = None,
    ):

        self.outcome_regressor = outcome_regressor
        self.sample_weight_regressor = sample_weight_regressor
        super().__init__()

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame, y: pd.DataFrame):
        """Fit the outcome model to the data.

        The treatment vector is concatenated to X before passing the inputs to
        outcome regressor

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        t : pd.DataFrame
            Treatment variable.
        y : pd.DataFrame
            Target variable.

        Returns
        -------
        self
            The object itself
        """
        if self.sample_weight_regressor is not None:
            self.sample_weight_regressor.fit(X, t)
            self.weights_ = predict_inverse_density_weight(
                self.sample_weight_regressor,
                X,
                t,
            ).reshape(-1)
        else:
            self.weights_ = None

        self.fit_kwargs_ = self._prepare_fit_kwargs(X, t, y, self.weights_)
        self.outcome_regressor.fit(**self.fit_kwargs_)
        return self

    def _prepare_fit_kwargs(
        self,
        X: pd.DataFrame,
        t: pd.DataFrame,
        y: pd.DataFrame,
        weights: np.ndarray,
    ) -> dict:

        dataset = {"X": self._prepare_input_array(X, t), "y": y}

        if weights is not None:
            kwarg_name = "sample_weight"
            if isinstance(self.outcome_regressor, Pipeline):
                kwarg_name = self.outcome_regressor.steps[-1][0] + "__sample_weight"
            dataset[kwarg_name] = weights
        return dataset

    def _get_n_samples(self, value) -> int:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return len(value)
        return super()._get_n_samples(value)

    def _prepare_input_array(
        self,
        X: pd.DataFrame,
        t: pd.DataFrame,
    ) -> pd.DataFrame:
        """Handles how to use X and t as input.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        t : pd.DataFrame
            Treatment variable.

        Returns
        -------
        pd.DataFrame
            DataFrame to be passed to outcome regressor
        """
        return self._concat(X, t)

    def _concat(self, X, t):
        return pd.concat(
            [X.reset_index(drop=True), t.reset_index(drop=True)],
            axis=1,
        )

    def _predict(self, X: pd.DataFrame, t: pd.DataFrame) -> list[float]:
        """Predict the Average Dose-Response Curve for a list of treatment values.

        Parameters
        ----------
        X : pd.DataFrame
            Input data
        t : pd.DataFrame
            Treatment values at which to evaluate the response.

        Returns
        -------
        list[float]
            List of predicted average treatment effects for each treatment value.
        """
        ys = []

        for i in range(t.shape[0]):
            repeated_t = pd.concat([t.iloc[[i]].copy()] * X.shape[0], ignore_index=True)
            ate = self.outcome_regressor.predict(
                self._prepare_input_array(X, repeated_t)
            ).mean()
            ys.append(ate)
        return ys

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sklearn.compose import ColumnTransformer, make_column_selector
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "encode_categorical",
                    OneHotEncoder(
                        drop="first",
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

        return [{"outcome_regressor": make_pipeline(preprocessor, LinearRegression())}]
