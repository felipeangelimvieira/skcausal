"""Base classes for ADRF estimations"""

import numpy as np
import polars as pl
from skbase.base import BaseEstimator as _BaseEstimator
from skcausal.datatypes import convert
from skcausal.base.mixin import TreatmentCheckMixin


class BaseAverageCausalResponseEstimator(TreatmentCheckMixin, _BaseEstimator):
    """Base class for ADRF Estimators.

    Subclasses must override `fit`, and `predict_individual` if the estimator
    has `capability:predicts_individual` tag set to True. Otherwise, they must override
    `fit` and `predict_average_treatment_effect`.

    If the method do not support individual predictions, the `predict_individual`
    method returns the same result as `predict_average_treatment_effect`, i.e., ignores
    the covariates.

    If the method support individual predictions, the `predict_average_treatment_effect`
    will, by default, call `predict_individual` and take the average of the results.
    Users may override for performance reasons.
    """

    _tags = {
        "backend": "polars",
        "capability:t_type": ["continuous", "categorical"],
        "capability:multidimensional_treatment": False,
    }

    def __init__(self): ...

    def fit(self, X, t, y):
        """
        Fit the estimator to the data.

        Public inputs are converted to the estimator backend before `_fit`
        is called.

        Parameters
        ----------
        X : DataFrame-like
            Input features.
        t : DataFrame-like
            Treatment variable.
        y : DataFrame-like
            Target variable.

        Returns
        -------
        self
            The object itself

        Raises
        ------
        ValueError
            If not implemented by the subclass.
        """
        X, t, y = self._check_and_transform(X, t, y, is_fit=True)

        self._fit(X=X, t=t, y=y)
        return self

    def _fit(self, X, t, y):
        """
        Fit the estimator to the data.

        Subclasses receive inputs already converted to the estimator backend.

        Parameters
        ----------
        X : backend-native dataframe
            Input features.
        t : backend-native dataframe
            Treatment variable.
        y : backend-native dataframe
            Target variable.

        Returns
        -------
        self
            The object itself

        Raises
        ------
        ValueError
            If not implemented by the subclass.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def predict(self, X, t):
        """
        Predict the average response for each treatment value in t.

        """

        X, t, _ = self._check_and_transform(X, t, y=None, is_fit=False)

        return np.array(self._predict(X, t))

    def _predict(self, X, t):
        """
        Predict using backend-native inputs.
        """

        raise NotImplementedError("This method must be implemented by subclasses.")

    def _check_and_transform_y(self, y, is_fit=False):
        y = convert(y, self.get_tag("backend"))
        return y

    def _check_and_transform_X(self, X, is_fit=False):
        X = convert(X, self.get_tag("backend"))
        return X

    def _get_n_samples(self, value) -> int:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                raise ValueError(
                    "Expected array-like input with at least one dimension."
                )
            return value.shape[0]
        if isinstance(value, pl.DataFrame):
            return value.height
        raise TypeError(
            f"Cannot infer number of samples from object of type {type(value).__name__}."
        )

    def _assert_same_number_of_samples(self, **kwargs) -> None:

        n_samples = [
            (name, self._get_n_samples(value))
            for name, value in kwargs.items()
            if value is not None
        ]

        if not all(n == n_samples[0][1] for _, n in n_samples):
            raise ValueError(
                "Inconsistent number of samples across inputs: "
                + ", ".join(f"{name} has {n} samples" for name, n in n_samples)
            )

    def _check_and_transform(self, X, t, y=None, is_fit=False):
        X = self._check_and_transform_X(X, is_fit=is_fit)
        t = self._check_and_transform_t(t, is_fit=is_fit)
        if y is not None:
            y = self._check_and_transform_y(y, is_fit=is_fit)

        self._assert_t_metadata_valid(metadata=self._t_metadata)
        self._assert_same_number_of_samples(X=X, t=t, y=y)

        return X, t, y
