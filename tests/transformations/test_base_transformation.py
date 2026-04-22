import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skcausal.transformations.base import BaseTransformation, SklearnBaseTransformation


class TransformationWithoutSoftDependencies(BaseTransformation):
    def _fit(self, X):
        return self

    def _transform(self, X):
        return X


class TransformationWithMissingSoftDependency(BaseTransformation):
    _tags = {
        "soft_dependencies": ["definitely_not_a_real_package_12345"],
    }

    def _fit(self, X):
        return self

    def _transform(self, X):
        return X


class NumpyReturningTransformation(BaseTransformation):
    def _fit(self, X):
        return self

    def _transform(self, X):
        return np.asarray(X.to_numpy())


def test_base_transformation_allows_empty_soft_dependencies():
    transformation = TransformationWithoutSoftDependencies()

    assert isinstance(transformation, TransformationWithoutSoftDependencies)


def test_base_transformation_checks_missing_soft_dependencies():
    with pytest.raises(
        ModuleNotFoundError, match="definitely_not_a_real_package_12345"
    ):
        TransformationWithMissingSoftDependency()


def test_fit_returns_self():
    transformation = TransformationWithoutSoftDependencies()
    X = pl.DataFrame({"x": [0.0, 1.0]})

    result = transformation.fit(X)

    assert result is transformation


@pytest.mark.parametrize(
    "backend,expected_type",
    [
        ("pandas", pd.DataFrame),
        ("polars", pl.DataFrame),
    ],
)
def test_backend_tag_converts_data_to_correct_type(backend, expected_type):
    class CapturingTransformation(BaseTransformation):
        _tags = {"backend": backend}

        def _fit(self, X):
            self._fit_X_type = type(X)
            return self

        def _transform(self, X):
            self._transform_X_type = type(X)
            return X

    transformation = CapturingTransformation()
    X = pl.DataFrame({"x": [0.0, 1.0]})

    transformation.fit(X)
    transformed = transformation.transform(X)

    assert transformation._fit_X_type == expected_type
    assert transformation._transform_X_type == expected_type
    assert isinstance(transformed, expected_type)


@pytest.mark.parametrize(
    "backend,expected_type",
    [
        ("pandas", pd.DataFrame),
        ("polars", pl.DataFrame),
    ],
)
def test_transform_coerces_numpy_output_to_backend(backend, expected_type):
    class BackendNumpyTransformation(NumpyReturningTransformation):
        _tags = {"backend": backend}

    transformation = BackendNumpyTransformation()
    X = pl.DataFrame({"x": [0.0, 1.0], "z": [1.0, 3.0]})

    transformation.fit(X)
    transformed = transformation.transform(X)

    assert isinstance(transformed, expected_type)

    if backend == "pandas":
        pd.testing.assert_frame_equal(transformed, X.to_pandas())
    else:
        assert transformed.equals(X)


class CapturingSklearnTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.fit_X_type_ = type(X)
        return self

    def transform(self, X):
        self.transform_X_type_ = type(X)
        return X.copy()


def test_sklearn_base_transformation_clones_and_fits_wrapped_transformer():
    transformer = StandardScaler()
    wrapped = SklearnBaseTransformation(transformer=transformer)
    X = pl.DataFrame({"x": [0.0, 1.0], "z": [1.0, 3.0]})

    wrapped.fit(X)

    assert wrapped.transformer_ is not transformer
    assert hasattr(wrapped.transformer_, "mean_")
    assert not hasattr(transformer, "mean_")


def test_sklearn_base_transformation_coerces_numpy_output_to_pandas():
    wrapped = SklearnBaseTransformation(transformer=StandardScaler())
    X = pl.DataFrame({"x": [0.0, 1.0], "z": [1.0, 3.0]})

    wrapped.fit(X)
    transformed = wrapped.transform(X)

    assert isinstance(transformed, pd.DataFrame)
    assert list(transformed.columns) == ["x", "z"]
    np.testing.assert_allclose(
        transformed.to_numpy(),
        wrapped.transformer_.transform(X.to_pandas()),
    )


def test_sklearn_base_transformation_uses_feature_names_for_expanded_output():
    wrapped = SklearnBaseTransformation(transformer=OneHotEncoder())
    X = pl.DataFrame(
        {
            "color": ["red", "blue", "red"],
            "shape": ["circle", "square", "triangle"],
        }
    )

    wrapped.fit(X)
    transformed = wrapped.transform(X)

    assert isinstance(transformed, pd.DataFrame)
    assert list(transformed.columns) == [
        "color_blue",
        "color_red",
        "shape_circle",
        "shape_square",
        "shape_triangle",
    ]
    np.testing.assert_allclose(
        transformed.to_numpy(),
        wrapped.transformer_.transform(X.to_pandas()).toarray(),
    )


def test_sklearn_base_transformation_delegates_fit_and_transform_with_pandas_input():
    transformer = CapturingSklearnTransformer()
    wrapped = SklearnBaseTransformation(transformer=transformer)
    X = pl.DataFrame({"x": [0.0, 1.0]})

    wrapped.fit(X)
    transformed = wrapped.transform(X)

    assert wrapped.transformer_.fit_X_type_ is pd.DataFrame
    assert wrapped.transformer_.transform_X_type_ is pd.DataFrame
    assert isinstance(transformed, pd.DataFrame)
    pd.testing.assert_frame_equal(transformed, X.to_pandas())
