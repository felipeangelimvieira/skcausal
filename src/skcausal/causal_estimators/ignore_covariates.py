import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator


class DirectNoCovariates(BaseAverageCausalResponseEstimator):
    """
    Predicts E[Y|T] directly, ignoring covariates X.

    This estimator should be used as baseline to compare against other
    causal estimators.


    Parameters
    ----------
    outcome_regressor : BaseEstimator
        Regressor to estimate the outcome
    random_state : int, default=0
        Random state for reproducibility.
    """

    _tags = {
        "capability:multidimensional_treatment": True,
        "backend": "pandas",
        "capability:t_type": ["continuous", "categorical"],
    }

    def __init__(
        self,
        outcome_regressor: BaseEstimator,
        random_state=0,
    ):
        self.outcome_regressor = outcome_regressor
        self.random_state = random_state

        super().__init__()

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame, y: pd.DataFrame):
        """Fits the DirectNoCovariates estimator.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.DataFrame
            Target variable.
        t : pd.DataFrame
            Treatment values.

        Returns
        -------
        self
            The object itself
        """

        self.outcome_regressor_ = clone(self.outcome_regressor)
        self.outcome_regressor_.fit(X=t, y=y)
        return self

    def _get_n_samples(self, value) -> int:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return len(value)
        return super()._get_n_samples(value)

    def _predict(self, X: pd.DataFrame, t: pd.DataFrame) -> list[float]:
        """
        Predict the average response for each treatment value in t.

        Parameters
        ----------
        X : pd.DataFrame
            The input data
        t : pd.DataFrame
            The treatment values

        Returns
        -------
        list[float]
            The average response for each treatment value in t.
        """

        effect = self.outcome_regressor_.predict(t)

        return effect.flatten().tolist()
