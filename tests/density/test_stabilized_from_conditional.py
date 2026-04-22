import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.neighbors import KernelDensity

from skcausal.density.base import BaseDensityEstimator
from skcausal.density.naive import NaiveDensityEstimator
from skcausal.density.stabilized_from_conditional import (
    KernelMarginalAndConditional,
    IntegratedMarginalAndConditional,
)


class GaussianLocationConditionalDensity(BaseDensityEstimator):
    _tags = {
        "backend": "pandas",
        "capability:t_type": ["continuous"],
        "density_kind": "conditional",
    }

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        super().__init__()

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame):
        return self

    def _predict_density(self, X: pd.DataFrame, t: pd.DataFrame) -> np.ndarray:
        mean = X.iloc[:, 0].to_numpy(dtype=float)
        values = t.iloc[:, 0].to_numpy(dtype=float)
        standardized = (values - mean) / self.scale
        density = np.exp(-0.5 * standardized**2) / (self.scale * np.sqrt(2.0 * np.pi))
        return density.reshape(-1, 1)


class UnitConditionalDensity(BaseDensityEstimator):
    _tags = {
        "backend": "pandas",
        "capability:t_type": ["continuous", "categorical"],
        "capability:multidimensional_treatment": True,
        "density_kind": "conditional",
    }

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame):
        return self

    def _predict_density(self, X: pd.DataFrame, t: pd.DataFrame) -> np.ndarray:
        return np.ones((len(t), 1), dtype=float)


def test_integrated_marginal_and_conditional_matches_empirical_average():
    X_train = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
    t_train = pl.DataFrame({"t": [0.0, 1.0, 2.0]})

    estimator = IntegratedMarginalAndConditional(
        conditional_density_estimator=GaussianLocationConditionalDensity(),
        n_samples=3,
    )
    estimator.fit(X_train, t_train)

    X_test = pl.DataFrame({"x": [0.0, 2.0]})
    t_test = pl.DataFrame({"t": [0.0, 1.0]})

    density_ratio = estimator.predict_density(X_test, t_test)

    def standard_normal_density(values):
        values = np.asarray(values, dtype=float)
        return np.exp(-0.5 * values**2) / np.sqrt(2.0 * np.pi)

    conditional_density = standard_normal_density([0.0, -1.0]).reshape(-1, 1)
    marginal_density = np.array(
        [
            standard_normal_density([0.0, -1.0, -2.0]).mean(),
            standard_normal_density([1.0, 0.0, -1.0]).mean(),
        ]
    ).reshape(-1, 1)
    expected = conditional_density / marginal_density

    np.testing.assert_allclose(density_ratio, expected)


def test_integrated_marginal_and_conditional_rejects_stabilized_estimators():
    with pytest.raises(ValueError, match="must predict a conditional density"):
        IntegratedMarginalAndConditional(
            conditional_density_estimator=NaiveDensityEstimator(
                density_kind="stabilized"
            )
        )


def test_integrated_marginal_and_conditional_propagates_wrapped_tags():
    estimator = IntegratedMarginalAndConditional(
        conditional_density_estimator=NaiveDensityEstimator(),
    )

    assert estimator.get_tag("capability:t_type") == ["continuous", "categorical"]
    assert estimator.get_tag("capability:multidimensional_treatment") is True
    assert estimator.get_tag("density_kind") == "stabilized"


def test_kernel_marginal_and_conditional_uses_groupwise_continuous_kdes():
    X_train = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    t_train = pl.DataFrame(
        {
            "t_cont": [0.0, 1.0, 10.0, 11.0],
            "t_cat": ["a", "a", "b", "b"],
        }
    )

    estimator = KernelMarginalAndConditional(
        conditional_density_estimator=UnitConditionalDensity(),
        kernel=KernelDensity(bandwidth=1.0),
    )
    estimator.fit(X_train, t_train)

    assert estimator.global_continuous_kernel_ is None
    assert set(estimator.category_kernels_) == {("a",), ("b",)}

    X_test = pl.DataFrame({"x": [10.0, 20.0]})
    t_test = pl.DataFrame(
        {
            "t_cont": [0.5, 10.5],
            "t_cat": ["a", "b"],
        }
    )

    density_ratio = estimator.predict_density(X_test, t_test)

    kernel_a = KernelDensity(bandwidth=1.0).fit(np.array([[0.0], [1.0]]))
    kernel_b = KernelDensity(bandwidth=1.0).fit(np.array([[10.0], [11.0]]))
    expected_marginal = np.array(
        [
            0.5 * np.exp(kernel_a.score_samples(np.array([[0.5]])))[0],
            0.5 * np.exp(kernel_b.score_samples(np.array([[10.5]])))[0],
        ]
    ).reshape(-1, 1)

    np.testing.assert_allclose(density_ratio, 1.0 / expected_marginal)


def test_kernel_marginal_and_conditional_uses_joint_categorical_mass_without_kde():
    X_train = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    t_train = pl.DataFrame(
        {
            "t_left": ["a", "a", "b", "b"],
            "t_right": ["x", "y", "x", "x"],
        }
    )

    estimator = KernelMarginalAndConditional(
        conditional_density_estimator=UnitConditionalDensity(),
        kernel=KernelDensity(bandwidth=1.0),
    )
    estimator.fit(X_train, t_train)

    X_test = pl.DataFrame({"x": [4.0, 5.0]})
    t_test = pl.DataFrame(
        {
            "t_left": ["a", "b"],
            "t_right": ["y", "x"],
        }
    )

    density_ratio = estimator.predict_density(X_test, t_test)
    expected_marginal = np.array([[0.25], [0.5]])

    np.testing.assert_allclose(density_ratio, 1.0 / expected_marginal)
