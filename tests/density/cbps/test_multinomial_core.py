import numpy as np
import pytest
from scipy.optimize import approx_fprime

from skcausal.density.cbps._design import DesignStandardizer
from skcausal.density.cbps._glm import fit_multinomial
from skcausal.density.cbps._optim import pinv_symmetric
from skcausal.density.cbps._multinomial import (
    bal_gradient,
    bal_loss,
    fit_cbps_multinomial,
    gmm_gradient,
    gmm_loss,
    gmm_weighting_matrix,
    multinomial_probs,
)


def simulate(n=600, J=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    coefs = np.array([[0.0, 1.0, -0.8], [0.0, -0.5, 0.9], [0.0, 0.3, 0.2]])[:, :J]
    logits = X @ coefs
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = (rng.uniform(size=n)[:, None] > probs.cumsum(axis=1)).sum(axis=1)
    T = np.eye(J)[labels]
    U = DesignStandardizer().fit(X).u_
    return U, T


@pytest.mark.parametrize("J", [2, 3, 4])
def test_gradients_match_finite_differences(J):
    U, T = simulate(J=J)
    rng = np.random.default_rng(1)
    params = fit_multinomial(U, T).ravel(order="F")
    params = params + rng.normal(scale=0.1, size=params.shape)
    inv_v = pinv_symmetric(gmm_weighting_matrix(U, multinomial_probs(U, params)))

    numeric = approx_fprime(params, lambda p: gmm_loss(p, U, T, inv_v), 1e-6)
    np.testing.assert_allclose(
        gmm_gradient(params, U, T, inv_v), numeric, rtol=1e-4, atol=1e-6
    )

    numeric_bal = approx_fprime(params, lambda p: bal_loss(p, U, T), 1e-6)
    np.testing.assert_allclose(
        bal_gradient(params, U, T), numeric_bal, rtol=1e-4, atol=1e-6
    )


@pytest.mark.parametrize("J", [2, 3])
@pytest.mark.parametrize("twostep", [True, False])
def test_exact_method_balances_covariate_means(J, twostep):
    U, T = simulate(J=J)

    result = fit_cbps_multinomial(U, T, method="exact", twostep=twostep)

    weighted_means = []
    for j in range(J):
        w = result.weights * T[:, j]
        weighted_means.append((w[:, None] * U).sum(axis=0) / w.sum())
    for j in range(1, J):
        np.testing.assert_allclose(weighted_means[j], weighted_means[0], atol=1e-3)
    assert result.converged


@pytest.mark.parametrize("method", ["over", "exact"])
def test_result_shapes_and_normalisation(method):
    U, T = simulate(J=3)

    result = fit_cbps_multinomial(U, T, method=method)

    assert result.beta.shape == (U.shape[1], 2)
    assert result.fitted_values.shape == T.shape
    np.testing.assert_allclose(result.fitted_values.sum(axis=1), 1.0)
    for j in range(3):
        np.testing.assert_allclose((result.weights * T[:, j]).sum(), 1.0)
    assert result.J >= 0 and result.mle_J >= 0


def test_over_method_improves_gmm_loss_relative_to_mle():
    U, T = simulate(J=3)
    result = fit_cbps_multinomial(U, T, method="over", twostep=True)
    assert result.J <= result.mle_J + 1e-12


def test_unstandardized_weights_are_inverse_probabilities():
    U, T = simulate(J=2)
    result = fit_cbps_multinomial(U, T, method="over", standardize=False)
    expected = (T / result.fitted_values).sum(axis=1)
    np.testing.assert_allclose(result.weights, expected)
