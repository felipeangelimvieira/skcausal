import numpy as np
import pytest

from skcausal.density.cbps._design import DesignStandardizer


def _make_X(n=50, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, k)) * np.array([1.0, 5.0, 0.1]) + 2.0


def test_transform_on_training_data_returns_orthonormal_u():
    X = _make_X()
    design = DesignStandardizer().fit(X)

    U = design.transform(X)

    np.testing.assert_allclose(U, design.u_)
    np.testing.assert_allclose(U.T @ U, np.eye(X.shape[1] + 1), atol=1e-10)
    assert design.k == X.shape[1] + 1


def test_coef_to_original_reproduces_linear_predictor_on_new_data():
    X = _make_X()
    X_new = _make_X(n=20, seed=1)
    design = DesignStandardizer().fit(X)
    beta_u = np.array([[0.5, -1.0], [2.0, 0.3], [-0.7, 1.1], [0.1, 0.0]])

    beta_orig = design.coef_to_original(beta_u)

    X_new_aug = np.column_stack([np.ones(len(X_new)), X_new])
    np.testing.assert_allclose(
        X_new_aug @ beta_orig, design.transform(X_new) @ beta_u, atol=1e-10
    )
    assert beta_orig.shape == beta_u.shape


def test_coef_to_original_accepts_1d_coefficients():
    X = _make_X()
    design = DesignStandardizer().fit(X)
    beta_u = np.array([0.5, 2.0, -0.7, 0.1])

    beta_orig = design.coef_to_original(beta_u)

    assert beta_orig.shape == (4,)
    X_aug = np.column_stack([np.ones(len(X)), X])
    np.testing.assert_allclose(X_aug @ beta_orig, design.u_ @ beta_u, atol=1e-10)


def test_constant_column_raises():
    X = _make_X()
    X[:, 1] = 3.0
    with pytest.raises(ValueError, match="constant"):
        DesignStandardizer().fit(X)


def test_collinear_columns_raise():
    X = _make_X()
    X = np.column_stack([X, X[:, 0] + X[:, 1]])
    with pytest.raises(ValueError, match="full rank"):
        DesignStandardizer().fit(X)
