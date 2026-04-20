from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from skcausal.causal_estimators._density_utils import (
    predict_density_array,
    is_stabilized_density,
)
from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator

from skcausal.density.base import BaseDensityEstimator

__all__ = [
    "CategoricalDoublyRobust",
    "CategoricalInversePropensityWeighting",
    "CategoricalDirectMethod",
]


class CategoricalDoublyRobustMixin:
    """Shared implementation for categorical-treatment response estimators.

    The mixin treats each unique categorical treatment row as a discrete
    treatment level, including multi-column treatment combinations. It stores
    the observed levels from the training data, fits one outcome regressor per
    level when needed, and computes the density-based correction terms used by
    the public categorical direct-method, inverse-propensity-weighting, and
    doubly robust estimators.
    """

    _tags = {
        "backend": "pandas",
        "capability:t_type": ["categorical"],
        "capability:multidimensional_treatment": True,
    }

    def _requires_density_estimator(self) -> bool:
        return True

    def _requires_outcome_regressor(self) -> bool:
        return True

    def _get_n_samples(self, value) -> int:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return len(value)
        return super()._get_n_samples(value)

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame, y: pd.DataFrame):
        """Fit category-specific outcome models and the treatment model."""

        self._y_array = np.asarray(y.to_numpy(), dtype=float).reshape(-1)
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

        density_columns = []
        for level in self.treatment_levels_:
            level_t = constant_treatment_frame(self._t, value=level, n_rows=len(X))
            density_columns.append(
                predict_density_array(self.density_estimator_, X, level_t).reshape(-1)
            )

        density_matrix = np.column_stack(density_columns)

        self.density_matrix_ = self._coerce_density_kind(
            density_matrix=density_matrix,
            marginals=self.treatment_marginals_,
            target_kind=self.target_density_kind,
            eps=1e-8,
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
            training_outcome_prediction = self._predict_outcome_for_level(
                self._X,
                level,
            )
            bias_correction_term = self._predict_bias_correction_term(
                training_outcome_prediction,
                level,
            )

            estimates_by_level[level] = outcome_prediction.mean() + bias_correction_term

        return np.array(
            [estimates_by_level[level] for level in requested_t], dtype=float
        )

    def _predict_outcome_for_level(self, X: pd.DataFrame, level):
        outcome_prediction = np.asarray(
            self.outcome_regressors_[level].predict(X),
            dtype=float,
        ).reshape(-1)
        return outcome_prediction

    def _predict_bias_correction_term(self, outcome_prediction: np.ndarray, level):
        eps = 1e-8
        mask = self.t_mask_[level]
        index = self.treatment_levels_index_[level]
        if self.target_density_kind == "stabilized":
            # In this case, we mask T_i = t
            bias_correction_term = (
                self._y_array[mask] - outcome_prediction[mask]
            ) / np.clip(self.density_matrix_[mask, index], eps, None)
            bias_correction_term = bias_correction_term.mean()
        elif self.target_density_kind == "conditional":
            bias_correction_term = (
                mask
                * (self._y_array - outcome_prediction)
                / np.clip(self.density_matrix_[:, index], eps, None)
            )
            bias_correction_term = bias_correction_term.mean()
        else:
            raise ValueError(
                f"Unexpected target_density_kind value: {self.target_density_kind!r}. "
                "Expected 'stabilized' or 'conditional'."
            )
        return float(bias_correction_term)

    def _coerce_density_kind(
        self, density_matrix, marginals, target_kind="stabilized", eps=1e-8
    ):
        """
        Convert density predictions into conditional probabilities or stabilized ratios.
        """

        probability_matrix = density_to_probability_matrix(
            self.density_estimator_,
            density_matrix,
            marginals,
            eps=eps,
        )

        if target_kind == "conditional":
            return probability_matrix
        if target_kind == "stabilized":
            return probability_matrix / np.clip(
                np.asarray(marginals, dtype=float).reshape(1, -1),
                eps,
                None,
            )

        raise ValueError(
            f"Unexpected target_kind value: {target_kind!r}. Expected 'stabilized' or 'conditional'."
        )


class CategoricalDoublyRobust(
    CategoricalDoublyRobustMixin, BaseAverageCausalResponseEstimator
):
    """
    Estimate average potential outcomes for categorical treatment levels.

    Each distinct row in ``t`` is treated as one observed treatment level, so
    multi-column categorical treatments are handled by working with their joint
    combinations. During ``fit``, the estimator clones and fits one outcome
    regressor per observed level, fits a treatment density estimator, and stores
    the empirical marginal frequency of each level.

    During ``predict``, the estimator returns a plug-in estimate based on the
    supplied covariate sample ``X`` plus a bias-correction term computed on the
    training sample. The estimator is doubly robust in the usual sense: the
    average potential outcome estimate remains consistent if either the
    level-specific outcome regressors or the treatment density model are
    correctly specified.

    Only treatment levels observed during ``fit`` may be requested at
    prediction time.

    Parameters
    ----------
    density_estimator : BaseDensityEstimator
        Estimator used to score the treatment assignment mechanism for each
        observed treatment level. It may return either conditional treatment
        probabilities ``P(T=t | X)`` or stabilized ratios
        ``P(T=t | X) / P(T=t)``.
    outcome_regressor : BaseEstimator
        Regressor cloned once per observed treatment level and fit on the
        subset of training rows assigned to that level.
    target_density_kind : {"conditional", "stabilized"}, default="stabilized"
        Density scale used in the bias-correction term. ``"conditional"``
        applies inverse conditional-probability weighting,
        ``1 / P(T=t | X)``, while ``"stabilized"`` applies inverse stabilized
        weighting, ``P(T=t) / P(T=t | X)``.
    """

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

        self._validate_init_params()

        density_estimator_multidim_t = self.get_tag(
            "capability:multidimensional_treatment"
        )
        if self.density_estimator is not None:
            density_estimator_multidim_t = self.density_estimator.get_tag(
                "capability:multidimensional_treatment",
                density_estimator_multidim_t,
            )

        self.set_tags(
            **{"capability:multidimensional_treatment": density_estimator_multidim_t}
        )

    def _validate_init_params(self) -> None:
        if self.target_density_kind not in {"conditional", "stabilized"}:
            raise ValueError(
                "target_density_kind must be either 'conditional' or 'stabilized'."
            )

        if self._requires_density_estimator():
            if self.density_estimator is None:
                raise ValueError("density_estimator must be provided.")
            if not hasattr(self.density_estimator, "fit") or not hasattr(
                self.density_estimator, "predict_density"
            ):
                raise TypeError(
                    "density_estimator must define fit and predict_density methods."
                )

        if self._requires_outcome_regressor():
            if self.outcome_regressor is None:
                raise ValueError("outcome_regressor must be provided.")
            if not hasattr(self.outcome_regressor, "fit") or not hasattr(
                self.outcome_regressor, "predict"
            ):
                raise TypeError(
                    "outcome_regressor must define fit and predict methods."
                )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sklearn.dummy import DummyRegressor

        from skcausal.density.naive import NaiveDensityEstimator

        return [
            {
                "density_estimator": NaiveDensityEstimator(),
                "outcome_regressor": DummyRegressor(strategy="mean"),
            },
            {
                "density_estimator": NaiveDensityEstimator(),
                "outcome_regressor": DummyRegressor(strategy="mean"),
                "target_density_kind": "conditional",
            },
        ]


class CategoricalInversePropensityWeighting(
    CategoricalDoublyRobustMixin, BaseAverageCausalResponseEstimator
):
    """
    Estimate average potential outcomes using categorical inverse weighting.

    This estimator reuses the categorical doubly robust machinery with the
    outcome-model component fixed at zero, so predictions are driven entirely by
    the density-based correction term. Each distinct row in ``t`` is treated as
    one categorical treatment level, including multi-column treatment
    combinations.

    Only treatment levels observed during ``fit`` may be requested at
    prediction time.

    Parameters
    ----------
    density_estimator : BaseDensityEstimator
        Estimator used to score the treatment assignment mechanism for each
        observed treatment level.
    target_density_kind : {"conditional", "stabilized"}, default="stabilized"
        Weighting scale used in the correction term. ``"conditional"`` uses
        inverse conditional probabilities, ``1 / P(T=t | X)``, while
        ``"stabilized"`` uses inverse stabilized weights,
        ``P(T=t) / P(T=t | X)``.
    """

    def __init__(
        self,
        density_estimator: BaseDensityEstimator,
        target_density_kind: str = "stabilized",
    ):
        self.density_estimator = density_estimator
        self.target_density_kind = target_density_kind

        super().__init__()

    def _requires_outcome_regressor(self) -> bool:
        return False

    def _fit_outcome_regressors(self, X, t, y): ...

    def _predict_outcome_for_level(self, X, level):
        return np.zeros(X.shape[0], dtype=float)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from skcausal.density.naive import NaiveDensityEstimator

        return [
            {"density_estimator": NaiveDensityEstimator()},
            {
                "density_estimator": NaiveDensityEstimator(),
                "target_density_kind": "conditional",
            },
        ]


class CategoricalDirectMethod(
    CategoricalDoublyRobustMixin, BaseAverageCausalResponseEstimator
):
    """
    Estimate average potential outcomes using only outcome regression.

    Each distinct row in ``t`` is treated as one observed categorical treatment
    level, including multi-column treatment combinations. The estimator fits one
    outcome regressor per observed level and predicts the mean of that level's
    regression function over the supplied covariate sample ``X``.

    No density model or bias correction is used, so the estimate relies solely
    on correct specification of the outcome regressors. Only treatment levels
    observed during ``fit`` may be requested at prediction time.

    Parameters
    ----------
    outcome_regressor : BaseEstimator
        Regressor cloned once per observed treatment level and fit on the
        subset of training rows assigned to that level.
    """

    def __init__(self, outcome_regressor: BaseEstimator):

        self.outcome_regressor = outcome_regressor

        super().__init__()

    def _requires_density_estimator(self) -> bool:
        return False

    def _fit_density_regressor(self, X, t): ...

    def _predict_bias_correction_term(self, outcome_prediction, level):
        return 0.0

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sklearn.dummy import DummyRegressor

        return [{"outcome_regressor": DummyRegressor(strategy="mean")}]


def _normalize_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def coerce_categorical_treatment(
    t: pd.DataFrame,
    *,
    estimator_name: str,
    argument_name: str = "t",
) -> np.ndarray:
    """
    Return categorical treatment rows as hashable scalar or tuple keys.

    Coerce the treatment to numpy array of tuples or scalars.
    Estimator name and argument name are used for error messages.

    Parameters
    ----------
    t : pd.DataFrame
        Treatment DataFrame to coerce.
    estimator_name : str
        Name of the estimator for error messages.
    argument_name : str, default="t"
        Name of the argument for error messages.

    Returns
    -------
    np.ndarray
        Array of hashable treatment values, either as scalars (for single-column
        treatments) or tuples (for multi-column treatments).
    """

    if t.shape[1] == 0:
        raise ValueError(
            f"{estimator_name} requires {argument_name} to contain at least one treatment column."
        )

    normalized_columns = []
    for column_name in t.columns:
        series = t[column_name]
        if series.isna().any():
            raise ValueError(
                f"{estimator_name} requires {argument_name} to contain no missing values."
            )
        normalized_columns.append(
            [_normalize_scalar(value) for value in series.to_list()]
        )

    if t.shape[1] == 1:
        # If a single treatment column we take the single normalized column
        return np.asarray(normalized_columns[0], dtype=object)

    joint_values = np.empty(len(t), dtype=object)
    for index, row_values in enumerate(zip(*normalized_columns)):
        joint_values[index] = tuple(row_values)
    return joint_values


def get_treatment_levels(treatment_values: np.ndarray) -> list:
    """Return observed treatment levels in order of first appearance."""

    return pd.Series(treatment_values, dtype=object).drop_duplicates().tolist()


def treatment_value_mask(treatment_values: np.ndarray, level) -> np.ndarray:
    """Return an elementwise equality mask for scalar or tuple treatment levels."""

    return np.fromiter((value == level for value in treatment_values), dtype=bool)


def validate_requested_treatment_values(
    requested_values: np.ndarray,
    *,
    observed_levels: list,
    estimator_name: str,
    argument_name: str = "t",
) -> None:
    missing_levels = [
        value for value in requested_values if value not in observed_levels
    ]
    if missing_levels:
        raise ValueError(
            f"{estimator_name} received unseen treatment values in {argument_name}: "
            f"{missing_levels!r}. Expected values drawn from {observed_levels!r}."
        )


def constant_treatment_frame(
    reference_t: pd.DataFrame,
    *,
    value,
    n_rows: int,
) -> pd.DataFrame:
    """Create a constant treatment frame matching the reference columns."""

    if reference_t.shape[1] == 1:
        return pd.DataFrame({reference_t.columns[0]: [value] * n_rows})

    if not isinstance(value, tuple) or len(value) != reference_t.shape[1]:
        raise ValueError(
            "Expected a tuple of treatment values matching the reference columns "
            f"{reference_t.columns.tolist()}, but received {value!r}."
        )

    return pd.DataFrame(
        {
            column_name: [column_value] * n_rows
            for column_name, column_value in zip(reference_t.columns, value)
        }
    )


def density_to_probability_matrix(
    density_estimator,
    density_matrix: np.ndarray,
    marginals: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Convert conditional or stabilized density outputs into probabilities."""

    probabilities = np.asarray(density_matrix, dtype=float)
    if is_stabilized_density(density_estimator):
        probabilities = probabilities * np.asarray(marginals, dtype=float).reshape(
            1, -1
        )

    denom = np.clip(probabilities.sum(axis=1, keepdims=True), eps, None)
    return probabilities / denom
