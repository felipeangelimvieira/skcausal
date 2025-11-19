"""Base classes for ADRF estimations"""

import numpy as np
import polars as pl
from skbase.base import BaseEstimator as _BaseEstimator

from skcausal.utils.polars import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    assert_schema_equal,
    to_dummies,
)
from skcausal.utils.mtype import convert_mtype


class BaseAverageCausalResponseEstimator(_BaseEstimator):
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
        "capability:supports_multidimensional_treatment": False,
        "supported_t_dtypes": [pl.Enum, pl.Boolean, *INTEGER_DTYPES, *FLOAT_DTYPES],
        "t_inner_mtype": np.ndarray,
        "X_inner_mtype": np.ndarray,
        "y_inner_mtype": np.ndarray,
        "store_X": False,
        "one_hot_encode_enum_columns": True,
        "tests:core": True,
    }

    def __init__(self):

        self._t_schema = None
        self._t_preprocessed_schema = None

    def fit(self, X, y, t):
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

        raw_t_df = self._to_polars_dataframe(t, variable_name="t")

        if not self.get_tag("capability:supports_multidimensional_treatment", False):
            if raw_t_df.width > 1:
                raise ValueError(
                    "This estimator does not support multi-dimensional treatments. The treatment variable must be 1-dimensional."
                )

        self._t_schema = raw_t_df.schema
        self._check_treatment_dtypes(raw_t_df)

        processed_t_df = self._preprocess_treatment_dataframe(raw_t_df)

        X_inner = self._check_and_transform_X(X)
        y_inner = self._check_and_transform_y(y)
        t_inner = self._convert_treatment_to_inner(processed_t_df)

        self._validate_input_shapes(X_inner, y_inner, t_inner)

        if self.get_tag("store_X", False):
            self._X = X_inner

        self._fit(X_inner, y_inner, t_inner)

        return self

    def _fit(self, X, y, t):
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

    def predict_adrf(self, X, t):
        """
        Predict the average treatment effect for each treatment value in t.

        """

        X_inner = self._check_and_transform_X(X)
        _, _, t_inner = self._prepare_treatment_inputs(t, check_schema=True)

        return self._predict_adrf(X_inner, t_inner)

    def _predict_adrf(self, X, t):
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

    def _check_and_transform_X(self, X):
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

    def _check_and_transform_y(self, y):
        inner_mtype = self._resolve_inner_mtype("y_inner_mtype", np.ndarray)
        if inner_mtype is np.ndarray:
            if isinstance(y, np.ndarray):
                result = y
            try:
                result = convert_mtype(y, np.ndarray)
            except Exception:  # pragma: no cover - fallback for array-likes
                result = np.asarray(y)
        else:
            if isinstance(y, inner_mtype):
                result = y
            else:
                result = convert_mtype(y, inner_mtype)

        if not isinstance(result, inner_mtype):
            raise TypeError(
                f"Expected y to be converted to {inner_mtype}, got {type(result)} instead."
            )

        return result

    def _to_polars_dataframe(self, value, *, variable_name: str) -> pl.DataFrame:
        if isinstance(value, pl.DataFrame):
            return value
        convert_kwargs = {}
        if self._t_schema is not None and isinstance(value, np.ndarray):
            expected_columns = list(self._t_schema.keys())
            if value.ndim == 1 and len(expected_columns) == 1:
                convert_kwargs["column_names"] = expected_columns
            elif value.ndim == 2 and value.shape[1] == len(expected_columns):
                convert_kwargs["column_names"] = expected_columns
        try:
            return convert_mtype(value, pl.DataFrame, **convert_kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"{self.__class__.__name__} expected `{variable_name}` to be a polars.DataFrame or convertible to one, "
                f"but received {type(value).__name__}."
            ) from exc

    def _check_treatment_dtypes(self, t: pl.DataFrame) -> None:
        supported_dtypes = self.get_tag("supported_t_dtypes", [])
        for column_name, dtype in zip(t.columns, t.dtypes):
            if supported_dtypes and dtype not in supported_dtypes:
                raise ValueError(
                    f"Column '{column_name}' has dtype {dtype}, which is not supported. Expected dtypes: {supported_dtypes}."
                )

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

    def _validate_input_shapes(self, X, y, t) -> None:
        n_samples = self._get_n_samples(X)
        n_targets = self._get_n_samples(y)
        n_treatments = self._get_n_samples(t)

        if n_samples != n_targets or n_samples != n_treatments:
            raise ValueError(
                "Inconsistent number of samples: "
                f"X has {n_samples}, y has {n_targets}, t has {n_treatments}."
            )

    def _resolve_inner_mtype(self, tag_key: str, default):
        inner_mtype = self.get_tag(tag_key, default)
        if inner_mtype == "np.ndarray":
            return np.ndarray
        if inner_mtype in {"pl.DataFrame", "polars.DataFrame"}:
            return pl.DataFrame
        return inner_mtype

    def _assert_treatment_schema(self, t):
        if self._t_schema is None:
            return

        if not isinstance(t, pl.DataFrame):
            t = convert_mtype(t, pl.DataFrame)

        assert_schema_equal(t.schema, self._t_schema)
