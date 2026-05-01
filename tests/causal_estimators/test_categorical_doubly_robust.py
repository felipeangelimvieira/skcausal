import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from skcausal.causal_estimators.categorical import (
    CategoricalDirectMethod,
    CategoricalDoublyRobust,
    CategoricalInversePropensityWeighting,
)
from skcausal.density.naive import NaiveDensityEstimator


class OffsetMeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, offset: float = 0.0):
        self.offset = offset

    def fit(self, X, y):
        values = np.asarray(y, dtype=float).reshape(-1)
        self.constant_ = float(values.mean()) + float(self.offset)
        return self

    def predict(self, X):
        return np.full(len(X), self.constant_, dtype=float)


def _make_single_treatment_dataset():
    X_train = pd.DataFrame(
        {
            "x0": np.linspace(-1.0, 1.0, 8),
            "x1": np.arange(8, dtype=float),
        }
    )
    t_train = pd.DataFrame({"t": [False, False, True, True, False, False, True, True]})
    expected = {False: 1.5, True: 4.0}
    y_train = pd.DataFrame({"y": [expected[value] for value in t_train["t"]]})

    X_query = pd.DataFrame({"x0": [-0.25, 0.25], "x1": [10.0, 11.0]})
    t_query = pd.DataFrame({"t": [False, True]})
    return X_train, t_train, y_train, X_query, t_query, np.array([1.5, 4.0])


def _make_multidimensional_treatment_dataset():
    X_train = pd.DataFrame(
        {
            "x0": np.linspace(-1.0, 1.0, 8),
            "x1": np.arange(8, dtype=float),
        }
    )
    treatment_levels = [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]
    repeated_levels = treatment_levels * 2
    t_train = pd.DataFrame(repeated_levels, columns=["t0", "t1"])
    expected = {
        (False, False): 0.5,
        (False, True): 1.5,
        (True, False): 2.5,
        (True, True): 4.5,
    }
    y_train = pd.DataFrame({"y": [expected[level] for level in repeated_levels]})

    X_query = pd.DataFrame(
        {
            "x0": [-0.75, -0.25, 0.25, 0.75],
            "x1": [20.0, 21.0, 22.0, 23.0],
        }
    )
    t_query = pd.DataFrame(treatment_levels, columns=["t0", "t1"])
    return (
        X_train,
        t_train,
        y_train,
        X_query,
        t_query,
        np.array([0.5, 1.5, 2.5, 4.5]),
    )


@pytest.mark.parametrize(
    "dataset_builder",
    [_make_single_treatment_dataset, _make_multidimensional_treatment_dataset],
    ids=["single_treatment", "multidimensional_treatment"],
)
@pytest.mark.parametrize("target_density_kind", ["conditional", "stabilized"])
def test_categorical_doubly_robust_corrects_constant_outcome_bias(
    dataset_builder,
    target_density_kind,
):
    X_train, t_train, y_train, X_query, t_query, expected = dataset_builder()
    estimator = CategoricalDoublyRobust(
        density_estimator=NaiveDensityEstimator(),
        outcome_regressor=OffsetMeanRegressor(offset=0.75),
        target_density_kind=target_density_kind,
    )

    estimator.fit(X_train, t_train, y_train)
    prediction = estimator.predict(t_query, X=X_query)

    np.testing.assert_allclose(prediction, expected)


@pytest.mark.parametrize("target_density_kind", ["conditional", "stabilized"])
def test_categorical_inverse_propensity_weighting_matches_observed_level_means(
    target_density_kind,
):
    X_train, t_train, y_train, X_query, t_query, expected = (
        _make_single_treatment_dataset()
    )
    estimator = CategoricalInversePropensityWeighting(
        density_estimator=NaiveDensityEstimator(),
        target_density_kind=target_density_kind,
    )

    estimator.fit(X_train, t_train, y_train)
    prediction = estimator.predict(t_query, X=X_query)

    np.testing.assert_allclose(prediction, expected)


def test_categorical_direct_method_fits_without_density_estimator():
    X_train, t_train, y_train, X_query, t_query, expected = (
        _make_single_treatment_dataset()
    )
    estimator = CategoricalDirectMethod(
        outcome_regressor=OffsetMeanRegressor(offset=0.0)
    )

    estimator.fit(X_train, t_train, y_train)
    prediction = estimator.predict(t_query, X=X_query)

    np.testing.assert_allclose(prediction, expected)


@pytest.mark.parametrize(
    "estimator_class",
    [
        CategoricalDoublyRobust,
        CategoricalInversePropensityWeighting,
        CategoricalDirectMethod,
    ],
)
def test_categorical_estimators_get_test_params_are_instantiable(estimator_class):
    test_params = estimator_class.get_test_params()
    if isinstance(test_params, dict):
        test_params = [test_params]

    assert test_params

    if estimator_class in {
        CategoricalDoublyRobust,
        CategoricalInversePropensityWeighting,
    }:
        density_kinds = {
            params.get("target_density_kind", "stabilized") for params in test_params
        }
        assert density_kinds == {"conditional", "stabilized"}

    for params in test_params:
        estimator = estimator_class(**params)
        assert isinstance(estimator, estimator_class)
