import numpy as np
import pytest

from skcausal.density.cbps._binary_att import (
    att_bal_gradient,
    att_bal_loss,
    att_gmm_gradient,
    att_gmm_loss,
    att_probs,
    att_weighting_matrix,
    fit_cbps_binary_att,
)
from skcausal.density.cbps._design import DesignStandardizer
from skcausal.density.cbps._glm import fit_logistic
from skcausal.density.cbps._optim import pinv_symmetric


def simulate(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    treat = (X @ [1.0, -0.5, 0.3] + rng.normal(size=n) > 0).astype(float)
    return DesignStandardizer().fit(X).u_, treat


def _central_difference(f, x, h=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        step = np.zeros_like(x)
        step[i] = h
        grad[i] = (f(x + step) - f(x - step)) / (2 * h)
    return grad


def test_gradients_match_finite_differences():
    U, treat = simulate()
    params = fit_logistic(U, treat) + 0.1
    inv_v = pinv_symmetric(att_weighting_matrix(U, treat, att_probs(U, params)))

    numeric = _central_difference(lambda p: att_gmm_loss(p, U, treat, inv_v), params)
    np.testing.assert_allclose(
        att_gmm_gradient(params, U, treat, inv_v), numeric, rtol=1e-5
    )

    numeric_bal = _central_difference(lambda p: att_bal_loss(p, U, treat), params)
    np.testing.assert_allclose(
        att_bal_gradient(params, U, treat), numeric_bal, rtol=1e-5
    )


@pytest.mark.parametrize("twostep", [True, False])
def test_exact_method_matches_control_means_to_treated_means(twostep):
    U, treat = simulate()

    result = fit_cbps_binary_att(U, treat, method="exact", twostep=twostep)

    treated = treat == 1
    treated_mean = U[treated].mean(axis=0)
    w = result.weights[~treated]
    control_mean = (w[:, None] * U[~treated]).sum(axis=0) / w.sum()
    np.testing.assert_allclose(control_mean, treated_mean, atol=1e-3)


@pytest.mark.parametrize("method", ["over", "exact"])
def test_weights_normalise_within_groups(method):
    U, treat = simulate()
    result = fit_cbps_binary_att(U, treat, method=method)

    assert result.fitted_values.shape == treat.shape
    assert result.beta.shape == (U.shape[1],)
    np.testing.assert_allclose(result.weights[treat == 1].sum(), 1.0)
    np.testing.assert_allclose(result.weights[treat == 0].sum(), 1.0)
    assert (result.weights > 0).all()


def test_unstandardized_weights_follow_r_definition():
    U, treat = simulate()
    result = fit_cbps_binary_att(U, treat, method="over", standardize=False)
    n, n_t = len(treat), treat.sum()
    p = result.fitted_values
    expected = np.where(treat == 1, n / n_t, n / n_t * p / (1 - p))
    np.testing.assert_allclose(result.weights, expected)
