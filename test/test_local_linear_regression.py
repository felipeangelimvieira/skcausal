import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from skcausal.sklearn.regression.local_linear_regression import (
    LocalLinearRegression,
)  # Update import path


# Fixtures
@pytest.fixture
def linear_data():
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 2])
    return X, y


@pytest.fixture
def constant_data():
    X = np.array([[0], [1], [2]])
    y = np.array([5, 5, 5])
    return X, y


@pytest.fixture
def multi_feature_data():
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
    return X, y


# Tests
def test_basic_prediction(linear_data):
    X, y = linear_data
    llr = LocalLinearRegression(length_scale=10.0)
    llr.fit(X, y)
    preds = llr.predict([[0.5], [1.5]])
    assert np.allclose(preds, [0.5, 1.5], rtol=0.1)


def test_constant_prediction(constant_data):
    X, y = constant_data
    llr = LocalLinearRegression(length_scale=0.1)
    llr.fit(X, y)
    preds = llr.predict([[0.5], [1.5]])
    assert np.allclose(preds, [5, 5])


def test_kernel_validation():
    with pytest.raises(ValueError):
        LocalLinearRegression(kernel="invalid_kernel")


def test_unfitted_error():
    llr = LocalLinearRegression()
    with pytest.raises(Exception):
        llr.predict([[0]])


def test_feature_mismatch(linear_data):
    X, y = linear_data
    llr = LocalLinearRegression().fit(X, y)
    with pytest.raises(ValueError):
        llr.predict([[0, 1]])


def test_zero_weights_behavior(constant_data):
    X, y = constant_data
    llr = LocalLinearRegression(length_scale=0.01, kernel="epanechnikov")
    llr.fit(X, y)
    preds = llr.predict([[10.0]])  # Far from training data
    assert np.isclose(preds[0], 5.0)


def test_large_length_scale_equivalence():
    X, y = make_regression(n_samples=100, n_features=2, noise=0, random_state=42)

    # Our model with large length_scale
    llr = LocalLinearRegression(length_scale=1e6)
    llr.fit(X, y)
    llr_preds = llr.predict(X)

    # Standard linear regression
    lr = LinearRegression().fit(X, y)
    lr_preds = lr.predict(X)

    assert np.allclose(llr_preds, lr_preds, atol=1e-3)


def test_multi_feature_support(multi_feature_data):
    X, y = multi_feature_data
    llr = LocalLinearRegression(length_scale=1.0)
    llr.fit(X, y)
    preds = llr.predict(X[:5])
    assert preds.shape == (5,)


def test_epanechnikov_kernel():
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 2])
    llr = LocalLinearRegression(length_scale=1.0, kernel="epanechnikov")
    llr.fit(X, y)
    preds = llr.predict([[0.5]])
    assert 0.4 <= preds[0] <= 0.6


def test_parameter_persistence():
    llr = LocalLinearRegression(length_scale=2.5, kernel="epanechnikov")
    assert llr.length_scale == 2.5
    assert llr.kernel == "epanechnikov"


def test_singular_matrix_handling():
    # Test case where XWX matrix is singular
    X = np.array([[0], [0], [1]])
    y = np.array([0, 0, 1])
    llr = LocalLinearRegression(length_scale=1.0)
    llr.fit(X, y)
    preds = llr.predict([[0.0]])
    assert 0 <= preds[0] <= 0.5  # Should handle via lstsq fallback
