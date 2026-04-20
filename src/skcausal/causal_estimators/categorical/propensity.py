import numpy as np
import pandas as pd

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
    "BinaryPropensityWeighting",
]


class BinaryPropensityWeighting(BaseAverageCausalResponseEstimator):
    """
    Propensity weighting estimator for a single categorical treatment variable.

    Parameters
    ----------
    treatment_regressor : BaseDensityEstimator
        Density estimator used to estimate treatment propensities.
    """

    _tags = {
        "backend": "pandas",
        "capability:t_type": ["categorical"],
        "capability:multidimensional_treatment": True,
        "one_hot_encode_enum_columns": False,
    }

    def __init__(
        self,
        treatment_regressor: BaseDensityEstimator,
    ):
        self.treatment_regressor = treatment_regressor

        super().__init__()

    def _get_n_samples(self, value) -> int:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return len(value)
        return super()._get_n_samples(value)

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame, y: pd.DataFrame):
        """Fit the treatment model and cache observed treatment levels."""

        self._X = X
        self._t = t
        self._y = np.asarray(y, dtype=float).reshape(-1)
        self.observed_treatment_values_ = coerce_categorical_treatment(
            t,
            estimator_name=self.__class__.__name__,
            argument_name="t",
        )
        self.treatment_levels_ = get_treatment_levels(self.observed_treatment_values_)

        if len(self.treatment_levels_) < 2:
            raise ValueError(
                f"{self.__class__.__name__} requires at least two observed treatment levels in the training data."
            )

        self.treatment_marginals_ = np.asarray(
            [
                float(
                    np.mean(
                        treatment_value_mask(self.observed_treatment_values_, level)
                    )
                )
                for level in self.treatment_levels_
            ],
            dtype=float,
        )

        self.treatment_regressor_ = self.treatment_regressor
        if self.treatment_regressor is not None:
            self.treatment_regressor_ = self.treatment_regressor.clone()
            self.treatment_regressor_.fit(X, t)
        return self

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

        return np.array(self._predict(X, t)).reshape((-1, 1)).mean()

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

        density_columns = []
        for level in self.treatment_levels_:
            level_t = constant_treatment_frame(self._t, value=level, n_rows=len(X))
            density_columns.append(
                predict_density_array(self.treatment_regressor_, X, level_t).reshape(-1)
            )

        density_matrix = np.column_stack(density_columns)

        is_stabilized = is_stabilized_density(self.treatment_regressor_)
        probability_matrix = None
        if not is_stabilized:
            probability_matrix = density_to_probability_matrix(
                self.treatment_regressor_,
                density_matrix,
                self.treatment_marginals_,
            )

        estimates_by_level = {}
        for index, level in enumerate(self.treatment_levels_):
            mask = treatment_value_mask(self.observed_treatment_values_, level)

            if is_stabilized:
                estimates_by_level[level] = float(
                    np.mean(
                        self._y[mask] / np.clip(density_matrix[mask, index], 1e-8, None)
                    )
                )
                continue

            estimates_by_level[level] = float(
                np.mean(
                    self._y[mask] / np.clip(probability_matrix[mask, index], 1e-8, None)
                )
                * self.treatment_marginals_[index]
            )

        return [estimates_by_level[level] for level in requested_t]
