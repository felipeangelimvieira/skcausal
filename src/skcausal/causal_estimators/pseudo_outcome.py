from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator

from skcausal.causal_estimators._density_utils import predict_density_array
from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.density.base import BaseDensityEstimator

__all__ = ["DoublyRobustPseudoOutcome"]


class DoublyRobustPseudoOutcome(BaseAverageCausalResponseEstimator):
    """
    Doubly-Robust pseudo-outcome for a single treatment variable.

    Based on the work of [1]

    Steps:
        1) Fit outcome model $\hat{\mu}(x, t)$ on (X, T) -> Y.
        2) Fit a treatment density model that outputs P(T|X)/P(T).
        3) For each i, compute:
            - stabilized weight $w_i = \hat{\pi}(T_i) / \hat{\pi}(T_i|X_i)$
            - \hat{\mu}_T(T_i) = E_X[ \hat{\mu}(X, T_i) ]      (estimated by averaging μ_hat(X, T_i) over X)
            - ξ_i = (Y_i - \hat{\mu}(X_i,T_i)) * w_i  + \hat{\mu}_T(T_i)
    4) Regress ξ_i on T_i with a final regressor to get m(t) = E[Y^t].

    Optionally, pass a n_pseudo_samples to avoid computing over all t_i

    Parameters
    ----------
    density_estimator : BaseDensityEstimator
        Density estimator used to estimate the treatment density ratio P(T|X)/P(T).
        Must have density_kind='stabilized'.
    outcome_regressor : BaseEstimator
        Sklearn-like regressor to use for estimating the outcome model μ(x, t).
    pseudo_outcome_regressor : BaseEstimator
        Sklearn-like regressor to use for estimating the final pseudo-outcome smoother.
    n_pseudo_samples : int, optional
        Number of samples to use for computing the pseudo-outcomes (without
        replacement).
        If None, uses all samples

    [1] Kennedy, Edward H., et al. "Non-parametric methods for doubly robust
      estimation of continuous treatment effects." Journal of the Royal
      Statistical Society Series B: Statistical Methodology 79.4 (2017):
      1229-1245.
    """

    _tags = {
        "backend": "pandas",
        "capability:t_type": ["continuous", "categorical"],
        "capability:multidimensional_treatment": False,
    }

    def __init__(
        self,
        density_estimator: BaseDensityEstimator,
        outcome_regressor: BaseEstimator,
        pseudo_outcome_regressor: BaseEstimator,
        n_pseudo_samples: Optional[int] = None,
        random_state: int = 42,
    ):
        self.density_estimator = density_estimator
        self.outcome_regressor = outcome_regressor
        self.pseudo_outcome_regressor = pseudo_outcome_regressor
        self.n_pseudo_samples = n_pseudo_samples
        self.random_state = random_state
        super().__init__()

        self._validate_init_params()

        self.set_tags(
            **{
                "capability:t_type": self.density_estimator.get_tag(
                    "capability:t_type",
                    self.get_tag("capability:t_type"),
                ),
                "capability:multidimensional_treatment": self.density_estimator.get_tag(
                    "capability:multidimensional_treatment",
                    self.get_tag("capability:multidimensional_treatment"),
                ),
            }
        )

    def _validate_init_params(self) -> None:
        if self.density_estimator is None:
            raise ValueError("density_estimator must be provided.")
        if not hasattr(self.density_estimator, "fit") or not hasattr(
            self.density_estimator, "predict_density"
        ):
            raise TypeError(
                "density_estimator must define fit and predict_density methods."
            )

        for name, regressor in (
            ("outcome_regressor", self.outcome_regressor),
            ("pseudo_outcome_regressor", self.pseudo_outcome_regressor),
        ):
            if regressor is None:
                raise ValueError(f"{name} must be provided.")
            if not hasattr(regressor, "fit") or not hasattr(regressor, "predict"):
                raise TypeError(f"{name} must define fit and predict methods.")

        if self.density_estimator.get_tag("density_kind") != "stabilized":
            raise ValueError(
                "density_estimator must have density_kind='stabilized'. Consider "
                "wrapping it with KernelMarginalAndConditional or "
                "IntegratedMarginalAndConditional."
            )

        if self.n_pseudo_samples is not None:
            if isinstance(self.n_pseudo_samples, bool) or not isinstance(
                self.n_pseudo_samples, (int, np.integer)
            ):
                raise TypeError("n_pseudo_samples must be an integer or None.")
            if int(self.n_pseudo_samples) < 1:
                raise ValueError("n_pseudo_samples must be greater than or equal to 1.")
            self.n_pseudo_samples = int(self.n_pseudo_samples)

    @staticmethod
    def _as_1d(value, name: str) -> np.ndarray:
        if isinstance(value, pl.DataFrame):
            array = value.to_numpy()
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            array = value.to_numpy()
        else:
            array = np.asarray(value)

        if array.ndim == 0:
            raise ValueError(f"{name} must be array-like with at least one element.")
        if array.ndim == 1:
            return np.asarray(array).reshape(-1)
        if array.ndim == 2 and array.shape[1] == 1:
            return np.asarray(array).reshape(-1)
        raise ValueError(f"{name} must be one-dimensional or single-column.")

    def _get_n_samples(self, value) -> int:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return len(value)
        return super()._get_n_samples(value)

    @staticmethod
    def _concat_features(X: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
        if len(X) != len(t):
            raise ValueError(
                "X and t must contain the same number of rows when building the "
                "outcome-regression features."
            )
        return pd.concat(
            [X.reset_index(drop=True), t.reset_index(drop=True)],
            axis=1,
        )

    @staticmethod
    def _repeat_treatment_row(
        t: pd.DataFrame, row_index: int, n_rows: int
    ) -> pd.DataFrame:
        return pd.concat([t.iloc[[row_index]].copy()] * n_rows, ignore_index=True)

    def _select_anchor_indices(self, n_samples: int) -> np.ndarray:
        if self.n_pseudo_samples is None:
            return np.arange(n_samples, dtype=int)
        if self.n_pseudo_samples > n_samples:
            raise ValueError(
                "n_pseudo_samples cannot be greater than the number of training samples."
            )
        rng = np.random.default_rng(self.random_state)
        selected = rng.choice(n_samples, size=self.n_pseudo_samples, replace=False)
        return np.sort(selected.astype(int))

    def _predict_outcome(self, X: pd.DataFrame, t: pd.DataFrame) -> np.ndarray:
        design_matrix = self._concat_features(X, t)
        predictions = self.outcome_regressor_.predict(design_matrix)
        return self._as_1d(predictions, name="outcome predictions")

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame, y: pd.DataFrame):
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("DoublyRobustPseudoOutcome requires at least one sample.")

        eps = 1e-8
        y_array = self._as_1d(y, name="outcome")

        self.density_estimator_ = self.density_estimator.clone()
        self.density_estimator_.fit(X, t)

        self.outcome_regressor_ = deepcopy(self.outcome_regressor)
        observed_design_matrix = self._concat_features(X, t)
        self.outcome_regressor_.fit(observed_design_matrix, y_array)

        observed_outcomes = self._predict_outcome(X, t)
        observed_density = predict_density_array(self.density_estimator_, X, t).reshape(
            -1
        )
        observed_density = np.clip(
            np.asarray(observed_density, dtype=float),
            eps,
            None,
        )

        self.anchor_indices_ = self._select_anchor_indices(n_samples)
        self.anchor_treatments_ = t.iloc[self.anchor_indices_].reset_index(drop=True)

        anchor_mean_outcomes = np.empty(self.anchor_indices_.shape[0], dtype=float)
        for position, row_index in enumerate(self.anchor_indices_):
            repeated_t = self._repeat_treatment_row(
                t, row_index=row_index, n_rows=n_samples
            )
            anchor_mean_outcomes[position] = self._predict_outcome(X, repeated_t).mean()

        anchor_residuals = (
            y_array[self.anchor_indices_] - observed_outcomes[self.anchor_indices_]
        )
        anchor_pseudo_outcomes = (
            anchor_residuals / observed_density[self.anchor_indices_]
            + anchor_mean_outcomes
        )

        self.pseudo_outcome_regressor_ = deepcopy(self.pseudo_outcome_regressor)
        self.pseudo_outcome_regressor_.fit(
            self.anchor_treatments_,
            anchor_pseudo_outcomes,
        )
        return self

    def _predict(self, t: pd.DataFrame, X: pd.DataFrame = None) -> np.ndarray:
        predictions = self.pseudo_outcome_regressor_.predict(t)

        if isinstance(predictions, pd.Series):
            return predictions.to_numpy().reshape(-1, 1)
        elif isinstance(predictions, pd.DataFrame):
            if predictions.shape[1] != 1:
                raise ValueError(
                    "Expected pseudo_outcome_regressor to predict a single column, "
                    f"but got {predictions.shape[1]} columns."
                )
            return predictions.to_numpy().reshape(-1, 1)
        else:
            return np.asarray(predictions, dtype=float).reshape(-1, 1)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sklearn.compose import ColumnTransformer, make_column_selector
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder
        from skcausal.density.naive import NaiveDensityEstimator

        def make_regressor_pipeline():
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
            return make_pipeline(preprocessor, LinearRegression())

        return [
            {
                "density_estimator": NaiveDensityEstimator("stabilized"),
                "outcome_regressor": make_regressor_pipeline(),
                "pseudo_outcome_regressor": make_regressor_pipeline(),
                "n_pseudo_samples": 10,
            }
        ]
