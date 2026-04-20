from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from skcausal.causal_estimators._density_utils import (
    is_stabilized_density,
    predict_density_array,
)
from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.causal_estimators.categorical._utils import (
    coerce_categorical_treatment,
    constant_treatment_frame,
    density_to_probability_matrix,
    get_treatment_levels,
    treatment_value_mask,
    validate_requested_treatment_values,
)
from skcausal.density.base import BaseDensityEstimator

__all__ = [
    "CategoricalDoublyRobust",
]


class CategoricalDoublyRobust(BaseAverageCausalResponseEstimator):
    """
    Doubly robust estimator for categorical treatment variables.

    For each categorical combination of treatment variables, this estimator fits
    * An outcome model to predict E[Y|X, T=t] using samples with T=t
    * A density model to estimate P(T|X) or P(T|X)/P(T)

    If the density model predicts P(T|X), P(T) is estimated as the empirical
    frequency of each treatment combination in the training data.

    Parameters
    ----------
    density_estimator : BaseDensityEstimator
        Density estimator used to estimate treatment propensities.
    outcome_regressor : BaseEstimator
        Regressor to estimate the outcome for each treatment combination.
    """

    _tags = {
        "backend": "pandas",
        "capability:t_type": ["categorical"],
        "capability:multidimensional_treatment": True,
    }

    def __init__(
        self,
        density_estimator: BaseDensityEstimator,
        outcome_regressor: BaseEstimator,
        target_density_kind: str = "stabilized",
    ):
        self.density_estimator = density_estimator
        self.outcome_regressor = outcome_regressor
        self.target_density_kind = target_density_kind

        super().__init__()

        density_estimator_multidim_t = self.density_estimator._get_tags(
            "capability:multidimensional_treatment"
        )
        self.set_tags(
            {"capability:multidimensional_treatment": density_estimator_multidim_t}
        )

    def _get_n_samples(self, value) -> int:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return len(value)
        return super()._get_n_samples(value)

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame, y: pd.DataFrame):
        """Fit category-specific outcome models and the treatment model."""

        self._y_array = y.values.ravel()
        self._X = X
        self._t = t

        # Coerce to np.array with hashable keys for
        # categorical treatment combinations
        observed_treatment_values_ = coerce_categorical_treatment(
            t,
            estimator_name=self.__class__.__name__,
            argument_name="t",
        )
        # Get unique treatment combinations observed in the training data
        self.treatment_levels_ = get_treatment_levels(observed_treatment_values_)
        self.treatment_levels_index_ = {
            level: index for index, level in enumerate(self.treatment_levels_)
        }

        self.t_mask_ = {}
        for level in self.treatment_levels_:
            self.t_mask_[level] = treatment_value_mask(
                observed_treatment_values_, level
            )

        if len(self.treatment_levels_) < 2:
            raise ValueError(
                f"{self.__class__.__name__} requires at least two observed "
                f"treatment levels in the training data."
            )

        self._fit_outcome_regressors(X, t, y)
        self._fit_density_regressor(X, t)

        return self

    def _fit_outcome_regressors(
        self, X: pd.DataFrame, t: pd.DataFrame, y: pd.DataFrame
    ):

        self.outcome_regressors_ = {}
        for level in self.treatment_levels_:
            mask = self.t_mask_[level]
            regressor = deepcopy(self.outcome_regressor)
            regressor.fit(X.loc[mask], y.loc[mask])
            self.outcome_regressors_[level] = regressor
        return

    def _fit_density_regressor(self, X: pd.DataFrame, t: pd.DataFrame):
        # We compute the marginal probability of each treatment level in the training data
        self.treatment_marginals_ = np.asarray(
            [float(np.mean(self.t_mask_[level])) for level in self.treatment_levels_],
            dtype=float,
        )

        # Verify if marginals sum up to 1
        marginal_sum = np.sum(self.treatment_marginals_)
        if not np.isclose(marginal_sum, 1.0):
            raise ValueError(
                f"Estimated treatment marginals sum to {marginal_sum:.4f},"
                f" which is not close to 1. "
                "Please report a bug"
            )

        self.density_estimator_ = self.density_estimator.clone()
        self.density_estimator_.fit(X, t)

        # Density matrix
        density_columns = []
        for level in self.treatment_levels_:
            level_t = constant_treatment_frame(self._t, value=level, n_rows=len(X))
            density_columns.append(
                predict_density_array(
                    self.treatment_regressor_, self._X, level_t
                ).reshape(-1)
            )

        density_matrix = np.column_stack(density_columns)
        eps = 1e-8

        self.density_matrix_ = self._coerce_density_kind(
            density_matrix=density_matrix,
            marginals=self.treatment_marginals_,
            target_kind=self.target_density_kind,
            eps=eps,
        )

    def _predict(self, X: pd.DataFrame, t: pd.DataFrame) -> list[float]:
        """
        Predict the average response for each treatment value in t.
        """

        requested_t = coerce_categorical_treatment(
            t,
            estimator_name=self.__class__.__name__,
            argument_name="t",
        )
        validate_requested_treatment_values(
            requested_t,
            observed_levels=self.treatment_levels_,
            estimator_name=self.__class__.__name__,
        )

        estimates_by_level = {}
        for level in self.treatment_levels_:

            outcome_prediction = self._predict_outcome_for_level(X, level)
            bias_correction_term = self._predict_bias_correction_term(
                outcome_prediction, level
            )

            estimates_by_level[level] = outcome_prediction.mean() + bias_correction_term

        return np.array([estimates_by_level[level] for level in requested_t])

    def _predict_outcome_for_level(self, X: pd.DataFrame, level):
        outcome_prediction = np.asarray(
            self.outcome_regressors_[level].predict(X),
            dtype=float,
        )
        return outcome_prediction

    def _predict_bias_correction_term(self, outcome_prediction: np.ndarray, level):
        mask = self.t_mask_[level]
        index = self.treatment_levels_index_[level]
        if self.target_density_kind == "stabilized":
            # In this case, we mask T_i = t
            bias_correction_term = (
                self._y_array[mask] - outcome_prediction[mask]
            ) / self.density_matrix_[mask, index]
            bias_correction_term = bias_correction_term.mean()
        elif self.target_density_kind == "conditional":
            bias_correction_term = (
                mask
                * (self._y_array - outcome_prediction)
                / self.density_matrix_[:, index]
            )
            bias_correction_term = bias_correction_term.mean()
        else:
            raise ValueError(
                f"Unexpected target_density_kind value: {self.target_density_kind!r}. "
                "Expected 'stabilized' or 'conditional'."
            )

    def _coerce_density_kind(
        self, density_matrix, marginals, target_kind="stabilized", eps=1e-8
    ):
        """
        Coerces density_matrix to the target_kind if the density regressor's density_kind tag does not match it.
        """

        density_estimator_kind = self.density_estimator.get_tag("density_kind")
        if density_estimator_kind == target_kind:
            return density_matrix
        elif target_kind == "stabilized":
            density_matrix = density_matrix / np.asarray(
                marginals, dtype=float
            ).reshape(1, -1)
        elif target_kind == "conditional":
            density_matrix = density_matrix * np.asarray(
                marginals, dtype=float
            ).reshape(1, -1)
        else:
            raise ValueError(
                f"Unexpected target_kind value: {target_kind!r}. Expected 'stabilized' or 'conditional'."
            )


class CategoricalInversePropensityWeighting(CategoricalDoublyRobust):
    """
    Inverse propensity weighting estimator for categorical treatment variables.

    This is a special case of the doubly robust estimator where the outcome model is set to predict a constant value for each treatment level.
    """

    def _fit_outcome_regressors(self, X, t, y): ...

    def _predict_outcome_for_level(self, X, level):
        return np.zeros(X.shape[0], dtype=float)


class CategoricalDirectMethod(CategoricalDoublyRobust):
    """
    Direct method estimator for categorical treatment variables.

    This is a special case of the doubly robust estimator where the density model is set to predict a constant value for each treatment level.
    """

    def _fit_density_regressor(self, X, t): ...

    def _predict_bias_correction_term(self, outcome_prediction, level):
        return 0.0
