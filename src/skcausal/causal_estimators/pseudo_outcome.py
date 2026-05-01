from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from skcausal.causal_estimators._density_utils import predict_density_array
from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.density.base import BaseDensityEstimator

__all__ = ["DoublyRobustPseudoOutcome"]


class _DummyFold:
    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X), dtype=int)
        yield indices, indices


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
    cv : int, optional
        Number of outer folds to use for nuisance-function out-of-fold
        predictions. Values of None, 0, or 1 disable the outer CV loop.
        When ``cross_fit`` is False, this preserves the current in-sample
        nuisance behavior.
    cross_fit : bool, default=False
        If True and ``cv`` is greater than 1, split each outer training fold
        into separate halves so the outcome nuisance and density nuisance are
        fitted on disjoint subsets before making out-of-fold predictions. When
        ``cv`` is None, 0, or 1, split the full dataset once into two nuisance
        training halves instead.

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
        cv: Optional[int] = 5,
        cross_fit: bool = False,
        n_pseudo_samples: Optional[int] = None,
        random_state: int = 42,
    ):
        self.density_estimator = density_estimator
        self.outcome_regressor = outcome_regressor
        self.pseudo_outcome_regressor = pseudo_outcome_regressor
        self.cv = cv
        self.cross_fit = cross_fit
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

        if self.cv is not None:
            if isinstance(self.cv, bool) or not isinstance(self.cv, (int, np.integer)):
                raise TypeError("cv must be an integer or None.")
            self.cv = int(self.cv)
            if self.cv < 0:
                raise ValueError("cv must be greater than or equal to 0 when provided.")

        if not isinstance(self.cross_fit, bool):
            raise TypeError("cross_fit must be a boolean.")

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

    def _get_outer_splitter(self):
        if self.cv is None or self.cv <= 1:
            return _DummyFold()
        return KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

    def _get_nuisance_splitter(self):
        if not self.cross_fit:
            return _DummyFold()
        return KFold(
            n_splits=2,
            shuffle=True,
            random_state=self.random_state,
        )

    def _split_nuisance_train_indices(
        self, train_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        train_idx = np.asarray(train_idx, dtype=int)
        if self.cross_fit and train_idx.shape[0] < 2:
            raise ValueError(
                "cross_fit=True requires at least 2 samples in each nuisance-training split."
            )

        nuisance_splitter = self._get_nuisance_splitter()
        outcome_positions, density_positions = next(nuisance_splitter.split(train_idx))
        return train_idx[outcome_positions], train_idx[density_positions]

    def _fit_nuisance_models(
        self, X: pd.DataFrame, t: pd.DataFrame, y_array: np.ndarray
    ) -> list[tuple[BaseDensityEstimator, BaseEstimator, np.ndarray]]:
        n_samples = X.shape[0]
        if self.cv is not None and self.cv > n_samples:
            raise ValueError(
                "cv cannot be greater than the number of training samples."
            )

        fitted_models: list[tuple[BaseDensityEstimator, BaseEstimator, np.ndarray]] = []

        for train_idx, test_idx in self._get_outer_splitter().split(X):
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)

            outcome_train_idx, density_train_idx = self._split_nuisance_train_indices(
                train_idx
            )

            X_outcome_train = X.iloc[outcome_train_idx]
            t_outcome_train = t.iloc[outcome_train_idx]
            X_density_train = X.iloc[density_train_idx]
            t_density_train = t.iloc[density_train_idx]

            density_model = self.density_estimator.clone()
            density_model.fit(X_density_train, t_density_train)

            outcome_model = deepcopy(self.outcome_regressor)
            outcome_model.fit(
                self._concat_features(X_outcome_train, t_outcome_train),
                y_array[outcome_train_idx],
            )

            fitted_models.append((density_model, outcome_model, test_idx))

        return fitted_models

    def _compute_observed_nuisance_predictions(
        self,
        X: pd.DataFrame,
        t: pd.DataFrame,
        fitted_models: list[tuple[BaseDensityEstimator, BaseEstimator, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return row-wise nuisance predictions from the held-out fold models."""
        n_samples = X.shape[0]
        observed_outcomes = np.empty(n_samples, dtype=float)
        observed_density = np.empty(n_samples, dtype=float)

        for density_model, outcome_model, prediction_indices in fitted_models:
            X_prediction = X.iloc[prediction_indices]
            t_prediction = t.iloc[prediction_indices]

            observed_outcomes[prediction_indices] = self._as_1d(
                outcome_model.predict(
                    self._concat_features(X_prediction, t_prediction)
                ),
                name="outcome predictions",
            )
            observed_density[prediction_indices] = predict_density_array(
                density_model,
                X_prediction,
                t_prediction,
            ).reshape(-1)

        return observed_outcomes, observed_density

    @staticmethod
    def _build_anchor_outcome_model_lookup(
        fitted_models: list[tuple[BaseDensityEstimator, BaseEstimator, np.ndarray]],
    ) -> dict[int, BaseEstimator]:
        """Map each held-out sample index to the outcome model that predicts it."""
        outcome_model_lookup: dict[int, BaseEstimator] = {}

        for _, outcome_model, prediction_indices in fitted_models:
            for prediction_index in prediction_indices:
                outcome_model_lookup[int(prediction_index)] = outcome_model

        return outcome_model_lookup

    def _compute_anchor_mean_outcomes(
        self,
        X: pd.DataFrame,
        t: pd.DataFrame,
        fitted_models: list[tuple[BaseDensityEstimator, BaseEstimator, np.ndarray]],
    ) -> np.ndarray:
        """Average each anchor treatment over X using its anchor-specific outcome model."""
        n_samples = X.shape[0]
        anchor_mean_outcomes = np.empty(self.anchor_indices_.shape[0], dtype=float)
        outcome_model_lookup = self._build_anchor_outcome_model_lookup(fitted_models)

        for position, row_index in enumerate(self.anchor_indices_):
            try:
                outcome_model = outcome_model_lookup[int(row_index)]
            except KeyError as error:
                raise ValueError(
                    "Each anchor must belong to one held-out prediction set when computing "
                    "anchor mean outcomes."
                ) from error
            repeated_t = self._repeat_treatment_row(
                t,
                row_index=row_index,
                n_rows=n_samples,
            )
            anchor_mean_outcomes[position] = self._as_1d(
                outcome_model.predict(self._concat_features(X, repeated_t)),
                name="anchor outcome predictions",
            ).mean()

        return anchor_mean_outcomes

    def _store_fitted_nuisance_models(
        self,
        fitted_models: list[tuple[BaseDensityEstimator, BaseEstimator, np.ndarray]],
    ) -> None:
        """Store nuisance fits from training without synthesizing a full-data refit."""
        self.nuisance_models_ = list(fitted_models)

        if len(self.nuisance_models_) == 1:
            self.density_estimator_, self.outcome_regressor_, _ = self.nuisance_models_[
                0
            ]
            return

        if hasattr(self, "density_estimator_"):
            delattr(self, "density_estimator_")
        if hasattr(self, "outcome_regressor_"):
            delattr(self, "outcome_regressor_")

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame, y: pd.DataFrame):
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("DoublyRobustPseudoOutcome requires at least one sample.")

        eps = 1e-8
        y_array = self._as_1d(y, name="outcome")

        fitted_nuisance_models = self._fit_nuisance_models(X, t, y_array)
        observed_outcomes, observed_density = (
            self._compute_observed_nuisance_predictions(
                X,
                t,
                fitted_nuisance_models,
            )
        )
        observed_density = np.clip(
            np.asarray(observed_density, dtype=float),
            eps,
            None,
        )

        self.anchor_indices_ = self._select_anchor_indices(n_samples)
        self.anchor_treatments_ = t.iloc[self.anchor_indices_].reset_index(drop=True)

        anchor_mean_outcomes = self._compute_anchor_mean_outcomes(
            X,
            t,
            fitted_nuisance_models,
        )

        anchor_residuals = (
            y_array[self.anchor_indices_] - observed_outcomes[self.anchor_indices_]
        )
        anchor_pseudo_outcomes = (
            anchor_residuals / observed_density[self.anchor_indices_]
            + anchor_mean_outcomes
        )

        self._store_fitted_nuisance_models(fitted_nuisance_models)

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
                "cv": 2,
                "cross_fit": True,
                "n_pseudo_samples": 10,
            }
        ]
