"""Compatibility tests against reference outputs of the R package ``CBPS``.

Fixtures live in ``fixtures/`` and were produced by ``generate_fixtures.R``
from the CSV files created by ``make_fixture_data.py``. The tests check that

1. our objective function evaluated at R's coefficients reproduces R's ``J``
   (the moment conditions and weighting matrix agree),
2. our optimum is at least as good as R's under our objective (the optimizer
   is not worse than R's ``optim``), and
3. fitted values, weights and coefficients agree within optimizer tolerance.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skcausal.density.cbps import CBPSCategorical, CBPSContinuous
from skcausal.density.cbps._binary_att import att_bal_loss, att_gmm_loss
from skcausal.density.cbps._continuous import (
    continuous_bal_loss,
    continuous_gmm_loss,
)
from skcausal.density.cbps._multinomial import bal_loss, gmm_loss

FIXTURES = Path(__file__).parent / "fixtures"
CASES = sorted(path.stem for path in FIXTURES.glob("*.json"))


def _load(case):
    with open(FIXTURES / f"{case}.json") as handle:
        fixture = json.load(handle)
    frame = pd.read_csv(FIXTURES / f"{fixture['data']}.csv")
    X = pl.DataFrame(frame.drop(columns="treat"))
    if fixture["data"] == "continuous":
        t = pl.DataFrame({"treat": frame["treat"].astype(float)})
    else:
        levels = sorted(frame["treat"].astype(str).unique())
        t = pl.DataFrame(
            {"treat": pl.Series(frame["treat"].astype(str)).cast(pl.Enum(levels))}
        )
    return fixture, X, t


def _fit(fixture, X, t):
    kwargs = dict(method=fixture["method"], twostep=fixture["twostep"])
    if fixture["data"] == "continuous":
        return CBPSContinuous(**kwargs).fit(X, t)
    return CBPSCategorical(att=fixture["ATT"], **kwargs).fit(X, t)


def _objectives_at(estimator, fixture):
    """Return (gmm_loss, bal_loss) of our objective at R's coefficients."""
    r_coef = np.asarray(fixture["coefficients"], dtype=float)
    result = estimator.result_
    inv_v = result.extra["inv_v_init"] if fixture["twostep"] else None

    if isinstance(estimator, CBPSContinuous):
        design = result.extra["design"]
        U = estimator.design_.u_
        linear_predictor = U @ estimator.design_.coef_from_original(r_coef[:, 0])
        beta_tilde = np.linalg.lstsq(
            design.xtilde_,
            (linear_predictor - design.t_mean_) / design.t_sd_,
            rcond=None,
        )[0]
        sigmasq_tilde = fixture["sigmasq"] / design.t_sd_**2
        params = np.concatenate([beta_tilde, [np.log(sigmasq_tilde)]])
        return (
            continuous_gmm_loss(params, design, inv_v),
            continuous_bal_loss(params, design),
        )

    U = estimator.design_.u_
    beta_u = estimator.design_.coef_from_original(r_coef)
    if fixture["ATT"] != 0:
        treat = (estimator._encode(estimator._t_for_test) == 1).astype(float)
        params = beta_u[:, 0]
        return att_gmm_loss(params, U, treat, inv_v), att_bal_loss(params, U, treat)
    codes = estimator._encode(estimator._t_for_test)
    T = np.eye(len(estimator.classes_))[codes]
    params = beta_u.ravel(order="F")
    return gmm_loss(params, U, T, inv_v), bal_loss(params, U, T)


def _require_comparable_solution(fixture):
    """Skip/xfail value comparisons where R's own solution is not reliable.

    * R's ``optim`` reports ``converged != 0`` when it stopped at ``maxit``
      without converging; the exact-method fixtures affected have balance
      losses orders of magnitude above ours.
    * For ATT with ``method="over"`` R's analytic gradient
      (``gmm.gradient`` in ``CBPSBinary.R``) scales the balance block by
      ``1/n_t`` instead of ``1/n``, so BFGS stops at a point that is not
      stationary for R's own objective; our optimum has a lower ``J`` under
      that objective (checked by ``test_our_optimum_is_at_least_as_good_as_r``).
    """
    if fixture["converged"] != 0:
        pytest.skip("R's optim did not converge for this fixture")
    if fixture["ATT"] != 0 and fixture["method"] == "over":
        pytest.xfail("R's ATT gradient is mis-scaled; solutions differ by design")


@pytest.fixture(scope="module", params=CASES)
def fitted_case(request):
    fixture, X, t = _load(request.param)
    estimator = _fit(fixture, X, t)
    estimator._t_for_test = t.to_pandas().iloc[:, 0]
    return request.param, fixture, estimator, X, t


def test_objective_at_r_coefficients_reproduces_r_J(fitted_case):
    name, fixture, estimator, _, _ = fitted_case
    if not fixture["twostep"]:
        pytest.skip(
            "continuous-updating J depends on the weighting matrix at the optimum"
        )
    gmm_at_r, bal_at_r = _objectives_at(estimator, fixture)
    continuous_exact = fixture["data"] == "continuous" and fixture["method"] == "exact"
    ours = bal_at_r if continuous_exact else gmm_at_r
    np.testing.assert_allclose(ours, fixture["J"], rtol=1e-3, atol=1e-9)


def test_our_optimum_is_at_least_as_good_as_r(fitted_case):
    name, fixture, estimator, _, _ = fitted_case
    gmm_at_r, bal_at_r = _objectives_at(estimator, fixture)
    if fixture["method"] == "exact":
        ours = estimator.result_.extra["bal_loss"]
        reference = bal_at_r
    else:
        ours = estimator.J_
        reference = gmm_at_r
    assert ours <= reference * (1 + 1e-3) + 1e-9, (ours, reference)


def test_fitted_values_match_r(fitted_case):
    name, fixture, estimator, _, _ = fitted_case
    _require_comparable_solution(fixture)
    r_fitted = np.asarray(fixture["fitted_values"], dtype=float)
    if isinstance(estimator, CBPSContinuous):
        ours = estimator.fitted_values_.reshape(-1, 1)
    elif len(estimator.classes_) == 2:
        ours = estimator.fitted_values_[:, [1]]
    else:
        ours = estimator.fitted_values_
    # Tolerances reflect optimizer noise on flat objectives (J agrees to 4
    # significant digits in every comparable case).
    np.testing.assert_allclose(ours, r_fitted, rtol=1e-2, atol=1e-2)


def test_weights_match_r(fitted_case):
    name, fixture, estimator, _, _ = fitted_case
    _require_comparable_solution(fixture)
    np.testing.assert_allclose(
        estimator.fitted_weights_,
        np.asarray(fixture["weights"], dtype=float),
        rtol=5e-2,
        atol=1e-5,
    )


def test_coefficients_match_r(fitted_case):
    name, fixture, estimator, _, _ = fitted_case
    _require_comparable_solution(fixture)
    r_coef = np.asarray(fixture["coefficients"], dtype=float)
    ours = np.asarray(estimator.coefficients_).reshape(r_coef.shape)
    np.testing.assert_allclose(ours, r_coef, rtol=2e-2, atol=2e-2)
    if isinstance(estimator, CBPSContinuous):
        np.testing.assert_allclose(estimator.sigmasq_, fixture["sigmasq"], rtol=1e-2)
