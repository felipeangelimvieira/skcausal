import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone

from skbase.base import BaseEstimator as _BaseEstimator
from skbase.utils.dependencies import _check_soft_dependencies

from skcausal.datatypes import convert
from skcausal.datatypes._typing import DataFrameLike


class BaseTransformation(_BaseEstimator):
    """Base class for data transformations.

    Tags
    ----
    backend : {"pandas", "polars"}
            Backend used by the protected implementation methods.
    soft_dependencies : list of str
            Optional dependencies checked at initialization time.
    """

    _tags = {
        "backend": "polars",
        "soft_dependencies": [],
    }

    def __init__(self):
        _check_soft_dependencies(*self.get_tag("soft_dependencies", []))
        super().__init__()

    def fit(self, X: DataFrameLike):
        """Fit the transformation to the data."""
        X = self._check_and_transform_X(X, is_fit=True)
        self._fit(X=X)
        return self

    def _fit(self, X):
        """Fit using backend-native input."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def transform(self, X: DataFrameLike):
        """Transform the data."""
        X = self._check_and_transform_X(X, is_fit=False)
        transformed = self._transform(X=X)
        return self._coerce_transform_output(transformed, reference_X=X)

    def _transform(self, X):
        """Transform using backend-native input."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def _check_and_transform_X(self, X: DataFrameLike, is_fit: bool = False):
        X = convert(X, self.get_tag("backend"))
        return X

    def _coerce_transform_output(self, transformed, *, reference_X: DataFrameLike):
        if isinstance(transformed, (pd.Series, pl.Series)):
            transformed = transformed.to_frame()

        if isinstance(transformed, (pd.DataFrame, pl.DataFrame)):
            return convert(transformed, self.get_tag("backend"))

        reference_X_pandas = convert(reference_X, "pandas")
        transformed_pandas = self._coerce_non_frame_output_to_pandas(
            transformed,
            reference_X=reference_X_pandas,
        )

        if self.get_tag("backend") == "pandas":
            return transformed_pandas

        return convert(transformed_pandas, self.get_tag("backend"))

    def _coerce_non_frame_output_to_pandas(
        self,
        transformed,
        *,
        reference_X: pd.DataFrame,
    ) -> pd.DataFrame:
        transformed_array = self._coerce_transform_output_to_2d_array(transformed)
        output_columns = self._get_output_columns(
            reference_X=reference_X,
            n_columns=transformed_array.shape[1],
        )

        if transformed_array.shape[0] == len(reference_X):
            return pd.DataFrame(
                transformed_array,
                columns=output_columns,
                index=reference_X.index,
            )

        return pd.DataFrame(transformed_array, columns=output_columns)

    @staticmethod
    def _coerce_transform_output_to_2d_array(transformed) -> np.ndarray:
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed_array = np.asarray(transformed)

        if transformed_array.ndim == 1:
            transformed_array = transformed_array.reshape(-1, 1)

        if transformed_array.ndim != 2:
            raise ValueError(
                "Expected transform output to be convertible to a 2D tabular "
                f"object, but received array with shape {transformed_array.shape}."
            )

        return transformed_array

    def _get_output_columns(
        self,
        *,
        reference_X: pd.DataFrame,
        n_columns: int,
    ) -> list[str]:
        if reference_X.shape[1] == n_columns:
            return list(reference_X.columns)

        return [f"x{i}" for i in range(n_columns)]


class SklearnBaseTransformation(BaseTransformation):
    """Adapter for sklearn-style transformations.

    Inputs are converted to the configured backend before being passed to the
    wrapped transformation. On fit, the wrapped transformation is cloned and the
    fitted clone is stored on ``transformer_``.

    Parameters
    ----------
    transformer : object
        A sklearn-compatible transformation implementing ``fit(X)`` and
        ``transform(X)``.
    """

    _tags = {
        "backend": "pandas",
        "soft_dependencies": ["sklearn"],
    }

    def __init__(self, transformer):
        self.transformer = transformer
        super().__init__()

    def _fit(self, X: pd.DataFrame):
        """Clone and fit the wrapped sklearn transformation."""
        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(X)
        return self

    def _transform(self, X: pd.DataFrame):
        """Delegate transformation to the fitted sklearn transformation."""
        return self.transformer_.transform(X)

    def _get_output_columns(
        self,
        *,
        reference_X: pd.DataFrame,
        n_columns: int,
    ) -> list[str]:
        get_feature_names_out = getattr(
            self.transformer_, "get_feature_names_out", None
        )
        if get_feature_names_out is not None:
            try:
                output_columns = list(get_feature_names_out(reference_X.columns))
            except (TypeError, ValueError):
                output_columns = list(get_feature_names_out())

            if len(output_columns) == n_columns:
                return output_columns

        return super()._get_output_columns(
            reference_X=reference_X,
            n_columns=n_columns,
        )
