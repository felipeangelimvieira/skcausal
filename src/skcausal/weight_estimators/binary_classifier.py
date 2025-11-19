import copy

import numpy as np
import polars as pl
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

from skcausal.utils.polars import INTEGER_DTYPES
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor

__all__ = ["BinaryClassifierWeightRegressor"]


class BinaryClassifierWeightRegressor(BaseBalancingWeightRegressor):
    """Compute balancing weights using a binary classification model.

    Parameters
    ----------
    classifier : ClassifierMixin
        Binary classifier that exposes :meth:`predict_proba`. A cloned copy is
        fitted internally for every call to :meth:`fit`.
    """

    _tags = {
        "t_inner_mtype": np.ndarray,
        "one_hot_encode_enum_columns": False,
        "supported_t_dtypes": [pl.Boolean, *INTEGER_DTYPES],
        "balancing_weight_type": "propensity_score",
    }

    def __init__(self, classifier: ClassifierMixin, random_state=0):
        self.random_state = random_state
        self.classifier = classifier
        super().__init__()

    def _fit(self, X: np.ndarray, t: np.ndarray):
        """Clone and train the underlying classifier on ``(X, t)``.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape ``(n_samples, n_features)``.
        t : np.ndarray
            Binary treatment assignments with shape ``(n_samples, 1)`` or
            ``(n_samples,)``.
        """
        self.classifier_ = copy.deepcopy(self.classifier)
        self.classifier_.fit(X=X, y=t)

    def _predict_sample_weight(self, X: np.ndarray, t: np.ndarray):
        """Return inverse propensity weights implied by the classifier.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape ``(n_samples, n_features)``.
        t : np.ndarray
            Binary treatment assignments with shape ``(n_samples, 1)`` or
            ``(n_samples,)``.

        Returns
        -------
        np.ndarray
            Column vector of inverse propensity weights with shape
            ``(n_samples, 1)``.
        """
        probas = self.classifier_.predict_proba(X)
        t_int = t.astype(int).flatten()
        p = probas[:, 0] * (1 - t_int) + probas[:, 1] * t_int

        return (1 / p).reshape(-1, 1)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"classifier": LogisticRegression()}]
