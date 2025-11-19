import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, clone

from skcausal.causal_estimators.base import BaseCausalResponseEstimator
from skcausal.utils.polars import convert_categorical_to_dummies


class DirectNoCovariates(BaseCausalResponseEstimator):
    """
    Predicts E[Y|T] directly, ignoring covariates X.


    Parameters
    ----------
    outcome_regressor : BaseEstimator
        Regressor to estimate the outcome
    random_state : int, default=0
        Random state for reproducibility.
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
        outcome_regressor: BaseEstimator,
        random_state=0,
    ):
        self.outcome_regressor = outcome_regressor
        self.random_state = random_state

        super().__init__()

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

        t = self._prepare_t(t)

        self.outcome_regressor_ = clone(self.outcome_regressor)
        self.outcome_regressor_.fit(X=t, y=y)

    def _prepare_t(self, t: pl.DataFrame) -> np.ndarray:
        """Prepare treatment values for prediction.

        Parameters
        ----------
        t : pl.DataFrame
            Treatment values.

        Returns
        -------
        np.ndarray
            Prepared treatment values.
        """

        t = convert_categorical_to_dummies(t)

        t = t.to_numpy().astype(np.float32)

        return t

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

        t = self._prepare_t(t)
        return self.outcome_regressor_.predict(t)

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
        t = self._prepare_t(t)
        effect = self.outcome_regressor_.predict(t)

        return effect.flatten().tolist()
