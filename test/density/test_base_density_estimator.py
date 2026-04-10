import numpy as np
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
