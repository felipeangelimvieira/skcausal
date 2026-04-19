"""Base classes for ADRF estimations"""

import numpy as np
import polars as pl
from skbase.base import BaseEstimator as _BaseEstimator
from skcausal.datatypes import (
    convert,
    collect_column_types,
)
from skcausal.utils.polars import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    assert_schema_equal,
    to_dummies,
)
from skcausal.utils.mtype import convert_mtype
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

        Abstract method that must be implemented by subclasses.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target variable.
        t : np.ndarray
            Treatment variable.

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

        Abstract method that must be implemented by subclasses.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target variable.
        t : np.ndarray
            Treatment variable.

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
        Predict the average treatment effect for each treatment value in t.

        """

        X, t, _ = self._check_and_transform(X, t, y=None, is_fit=False)

        return np.array(self._predict(X, t))

    def _predict(self, X, t):
        """
        Predict the average treatment effect for each treatment value in t.
        """

        raise NotImplementedError("This method must be implemented by subclasses.")

    def _preprocess_treatment_dataframe(self, t: pl.DataFrame) -> pl.DataFrame:
        """Apply deterministic preprocessing steps to the treatment dataframe."""

        if self._t_schema is not None:
            expected_columns = list(self._t_schema.keys())
            if list(t.columns) != expected_columns:
                t = t.select(expected_columns)

        if self.get_tag("one_hot_encode_enum_columns", False):
            for col, dtype in zip(t.columns, t.dtypes):
                if dtype == pl.Enum:
                    t = to_dummies(t, col)

        if self._t_preprocessed_schema is None:
            self._t_preprocessed_schema = t.schema
        else:
            assert_schema_equal(t.schema, self._t_preprocessed_schema)

        return t

    def _prepare_treatment_inputs(self, t, *, check_schema: bool):
        """Return raw, preprocessed, and inner representations of treatment input."""

        raw_df = self._to_polars_dataframe(t, variable_name="t")

        if check_schema:
            self._assert_treatment_schema(raw_df)

        processed_df = self._preprocess_treatment_dataframe(raw_df)
        t_inner = self._convert_treatment_to_inner(processed_df)

        return raw_df, processed_df, t_inner

    def _convert_treatment_to_inner(self, t: pl.DataFrame):
        """Convert treatment dataframe to the configured inner mtype."""

        inner_mtype = self._resolve_inner_mtype("t_inner_mtype", np.ndarray)
        if inner_mtype is np.ndarray:
            converted = convert_mtype(t, np.ndarray, dtype=np.float32)
        else:
            converted = convert_mtype(t, inner_mtype)

        if not isinstance(converted, inner_mtype):
            raise TypeError(
                f"Expected treatment to be converted to {inner_mtype}, got {type(converted)} instead."
            )

        return converted

    def _check_and_transform_X(self, X, is_fit=False):
        inner_mtype = self._resolve_inner_mtype("X_inner_mtype", np.ndarray)
        if inner_mtype is np.ndarray:
            if isinstance(X, np.ndarray):
                result = X
            try:
                result = convert_mtype(X, np.ndarray)
            except Exception:  # pragma: no cover - fallback for array-likes
                result = np.asarray(X)
        else:
            if isinstance(X, inner_mtype):
                result = X
            else:
                result = convert_mtype(X, inner_mtype)

        if not isinstance(result, inner_mtype):
            raise TypeError(
                f"Expected X to be converted to {inner_mtype}, got {type(result)} instead."
            )

        return result

    def _check_and_transform_y(self, y, is_fit=False):
        y = convert(y, self.get_tag("backend"))
        return y

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

    def _check_and_tranform(self, X, t, y=None, is_fit=False):
        X = self._check_and_transform_X(X, is_fit=is_fit)
        t = self._check_and_transform_t(t, is_fit=is_fit)
        if y is not None:
            y = self._check_and_transform_y(y, is_fit=is_fit)

        self._assert_t_metadata_valid(metadata=self._t_metadata)
        self._assert_same_number_of_samples(X=X, t=t, y=y)

        return X, t, y
