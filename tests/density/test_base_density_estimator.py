import numpy as np
import pandas as pd
import polars as pl
import pytest

from skcausal.density.base import BaseDensityEstimator


class DensityWithoutSoftDependencies(BaseDensityEstimator):
    def _fit(self, X: np.ndarray, t: np.ndarray):
        return self

    def _predict_density(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.ones((len(X), 1), dtype=float)


class DensityWithMissingSoftDependency(BaseDensityEstimator):
    _tags = {
        "soft_dependencies": ["definitely_not_a_real_package_12345"],
    }

    def _fit(self, X: np.ndarray, t: np.ndarray):
        return self

    def _predict_density(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.ones((len(X), 1), dtype=float)


class DensityWithMultidimensionalTreatmentCapability(BaseDensityEstimator):
    _tags = {
        "capability:multidimensional_treatment": True,
    }

    def _fit(self, X: np.ndarray, t: np.ndarray):
        return self

    def _predict_density(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.ones((len(X), 1), dtype=float)


class DensityWithRestrictedTTypes(BaseDensityEstimator):
    _tags = {
        "capability:t_type": ["continuous"],
    }

    def _fit(self, X: np.ndarray, t: np.ndarray):
        return self

    def _predict_density(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.ones((len(X), 1), dtype=float)


class DensityReturning1DOutput(BaseDensityEstimator):
    def _fit(self, X: np.ndarray, t: np.ndarray):
        return self

    def _predict_density(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.ones(len(X), dtype=float)


class DensityReturning3DOutput(BaseDensityEstimator):
    def _fit(self, X: np.ndarray, t: np.ndarray):
        return self

    def _predict_density(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.ones((len(X), 1, 1), dtype=float)


def test_base_density_estimator_allows_empty_soft_dependencies():
    estimator = DensityWithoutSoftDependencies()

    assert isinstance(estimator, DensityWithoutSoftDependencies)


def test_base_density_estimator_checks_missing_soft_dependencies():
    with pytest.raises(
        ModuleNotFoundError, match="definitely_not_a_real_package_12345"
    ):
        DensityWithMissingSoftDependency()


def test_base_density_estimator_rejects_multidimensional_treatment_by_default():
    estimator = DensityWithoutSoftDependencies()
    X = pl.DataFrame({"x": [0.0, 1.0]})
    t = pl.DataFrame({"t1": [0.1, 0.2], "t2": [0.3, 0.4]})

    with pytest.raises(ValueError, match="does not support multidimensional"):
        estimator.fit(X, t)


def test_base_density_estimator_allows_multidimensional_treatment_when_tagged():
    estimator = DensityWithMultidimensionalTreatmentCapability()
    X = pl.DataFrame({"x": [0.0, 1.0]})
    t = pl.DataFrame({"t1": [0.1, 0.2], "t2": [0.3, 0.4]})

    estimator.fit(X, t)

    assert isinstance(estimator, DensityWithMultidimensionalTreatmentCapability)


def test_predict_density_raises_when_t_schema_differs_from_fit():
    estimator = DensityWithoutSoftDependencies()
    X = pl.DataFrame({"x": [0.0, 1.0]})
    t_fit = pl.DataFrame({"t1": [0.1, 0.2]})
    t_predict = pl.DataFrame({"t2": [0.1, 0.2]})

    estimator.fit(X, t_fit)

    with pytest.raises(ValueError, match="does not match fit time"):
        estimator.predict_density(X, t_predict)


def test_predict_density_coerces_1d_output_to_column_vector():
    estimator = DensityReturning1DOutput()
    X = pl.DataFrame({"x": [0.0, 1.0]})
    t = pl.DataFrame({"t": [0.1, 0.2]})

    estimator.fit(X, t)
    density = estimator.predict_density(X, t)

    assert density.shape == (2, 1)
    np.testing.assert_allclose(density, np.ones((2, 1), dtype=float))


def test_predict_density_raises_when_private_output_is_not_2d():
    estimator = DensityReturning3DOutput()
    X = pl.DataFrame({"x": [0.0, 1.0]})
    t = pl.DataFrame({"t": [0.1, 0.2]})

    estimator.fit(X, t)

    with pytest.raises(ValueError, match="Expected density output to be 1D or 2D"):
        estimator.predict_density(X, t)


@pytest.mark.parametrize(
    "backend,expected_type",
    [
        ("pandas", pd.DataFrame),
        ("polars", pl.DataFrame),
    ],
)
def test_backend_tag_converts_data_to_correct_type(backend, expected_type):
    class CapturingDensity(BaseDensityEstimator):
        _tags = {"backend": backend}

        def _fit(self, X, t):
            self._fit_X_type = type(X)
            self._fit_t_type = type(t)
            return self

        def _predict_density(self, X, t):
            self._predict_X_type = type(X)
            self._predict_t_type = type(t)
            return np.ones((len(X), 1))

    estimator = CapturingDensity()
    X = pl.DataFrame({"x": [0.0, 1.0]})
    t = pl.DataFrame({"t1": [0.1, 0.2]})

    estimator.fit(X, t)
    estimator.predict_density(X, t)

    assert estimator._fit_X_type == expected_type
    assert estimator._fit_t_type == expected_type
    assert estimator._predict_X_type == expected_type
    assert estimator._predict_t_type == expected_type


def test_fit_raises_when_t_has_unsupported_column_type():
    estimator = DensityWithRestrictedTTypes()
    X = pl.DataFrame({"x": [0.0, 1.0]})
    t = pl.DataFrame({"t": ["a", "b"]})

    with pytest.raises(ValueError, match="not supported"):
        estimator.fit(X, t)
