import numpy as np
import pytest

from skcausal.density.cbps._continuous import (
    ContinuousDesign,
    continuous_bal_gradient,
    continuous_bal_loss,
    continuous_gmm_gradient,
    continuous_gmm_loss,
    continuous_weighting_matrix,
    fit_cbps_continuous,
)
from skcausal.density.cbps._design import DesignStandardizer
from skcausal.density.cbps._optim import pinv_symmetric


def simulate(n=600, seed=0, noise=0.8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = 2.0 + X @ [1.0, -0.5, 0.3] + rng.normal(scale=noise, size=n)
    return DesignStandardizer().fit(X).u_, t


def _central_difference(f, x, h=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        step = np.zeros_like(x)
        step[i] = h
        grad[i] = (f(x + step) - f(x - step)) / (2 * h)
    return grad


def test_continuous_design_whitens_and_standardizes():
    U, t = simulate()
    design = ContinuousDesign().fit(U, t)

    xt = design.xtilde_
    np.testing.assert_allclose(xt[:, 0], xt[0, 0])  # intercept column preserved
    np.testing.assert_allclose(xt[:, 1:].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cov(xt[:, 1:], rowvar=False), np.eye(3), atol=1e-10)
    np.testing.assert_allclose(design.ttilde_.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(design.ttilde_.std(ddof=1), 1.0)
    np.testing.assert_allclose(design.transform(U), xt)


def test_gradients_match_finite_differences():
    U, t = simulate()
    design = ContinuousDesign().fit(U, t)
    params = design.mle_params_ * 1.05
    inv_v = pinv_symmetric(continuous_weighting_matrix(params, design))

    numeric = _central_difference(lambda p: continuous_gmm_loss(p, design, inv_v), params)
    np.testing.assert_allclose(
        continuous_gmm_gradient(params, design, inv_v), numeric, rtol=1e-5, atol=1e-10
    )

    numeric_bal = _central_difference(lambda p: continuous_bal_loss(p, design), params)
    np.testing.assert_allclose(
        continuous_bal_gradient(params, design), numeric_bal, rtol=1e-5, atol=1e-10
    )


@pytest.mark.parametrize("twostep", [True, False])
def test_exact_method_removes_weighted_covariate_treatment_association(twostep):
    # Under strong confounding the just-identified system has no root (R's
    # CBPS reaches the same non-zero optimum); use moderate confounding here.
    U, t = simulate(noise=2.0)

    result = fit_cbps_continuous(U, t, method="exact", twostep=twostep)

    centered_u = U[:, 1:] - U[:, 1:].mean(axis=0)
    ttilde = (t - t.mean()) / t.std(ddof=1)
    association = (result.weights[:, None] * centered_u * ttilde[:, None]).sum(axis=0)
    np.testing.assert_allclose(association, 0.0, atol=1e-4)
    assert result.converged


@pytest.mark.parametrize("method", ["over", "exact"])
def test_result_contents(method):
    U, t = simulate()
    result = fit_cbps_continuous(U, t, method=method)

    assert result.beta.shape == (U.shape[1],)
    assert result.fitted_values.shape == t.shape
    np.testing.assert_allclose(result.weights.sum(), 1.0)
    assert result.extra["sigmasq"] > 0
    # U-space coefficients reproduce the conditional mean on the original scale.
    design = ContinuousDesign().fit(U, t)
    expected_mean = design.xtilde_ @ result.extra["beta_tilde"] * t.std(ddof=1) + t.mean()
    np.testing.assert_allclose(U @ result.beta, expected_mean, atol=1e-8)
    np.testing.assert_allclose(result.extra["sigmasq"], result.extra["sigmasq_tilde"] * t.var(ddof=1))


def test_over_improves_on_mle():
    U, t = simulate()
    result = fit_cbps_continuous(U, t, method="over")
    assert result.J <= result.mle_J + 1e-12
