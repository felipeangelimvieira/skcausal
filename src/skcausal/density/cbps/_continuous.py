"""CBPS for a continuous treatment (CBGPS; R's ``CBPS.Continuous``).

The treatment is modelled as ``t ~ N(x beta, sigma^2)`` after standardising
``t`` and whitening the non-constant design columns. With ``f(t | x)`` the
model density and ``f(t)`` the standard normal density on the standardised
scale, the moment conditions per unit are

* score:    ``x (t - x beta) / sigma^2``
* balance:  ``x * t * f(t) / f(t | x)``
* variance: ``(t - x beta)^2 / sigma^2 - 1``

Parameters are ``(beta, log sigma^2)``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm

from skcausal.density.cbps._glm import fit_linear
from skcausal.density.cbps._optim import (
    PROBS_MIN,
    minimize_bfgs,
    pick_best,
    pinv_symmetric,
    scale_search,
)
from skcausal.density.cbps._result import CBPSResult

__all__ = [
    "ContinuousDesign",
    "continuous_gmm_loss",
    "continuous_gmm_gradient",
    "continuous_bal_loss",
    "continuous_bal_gradient",
    "continuous_weighting_matrix",
    "fit_cbps_continuous",
]

_LOG_MIN = np.log(PROBS_MIN)
_LOG_MAX = np.log(1.0 - PROBS_MIN)


class ContinuousDesign:
    """Whitening / standardisation used by the continuous routine.

    Attributes
    ----------
    xtilde_ : ndarray of shape (n, k)
        Constant columns of ``U`` kept as-is, the remaining columns
        whitened with the Cholesky factor of their covariance and centred.
    ttilde_ : ndarray of shape (n,)
        Standardised treatment (``ddof=1``).
    stabilizers_ : ndarray of shape (n,)
        ``log f(ttilde)`` under the standard normal, clipped like R.
    mle_params_ : ndarray of shape (k + 1,)
        OLS coefficients followed by ``log sigma^2``.
    """

    def fit(self, U: np.ndarray, t: np.ndarray) -> "ContinuousDesign":
        U = np.asarray(U, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        n = U.shape[0]

        sd = U.std(axis=0, ddof=1)
        self.constant_columns_ = np.flatnonzero(sd <= 1e-10)
        self.varying_columns_ = np.flatnonzero(sd > 1e-10)

        varying = U[:, self.varying_columns_]
        chol_upper = np.linalg.cholesky(np.cov(varying, rowvar=False, ddof=1)).T
        self.whitener_ = np.linalg.inv(chol_upper)
        whitened = varying @ self.whitener_
        self.whitened_mean_ = whitened.mean(axis=0)

        self.t_mean_ = float(t.mean())
        self.t_sd_ = float(t.std(ddof=1))
        self.xtilde_ = self.transform(U)
        self.ttilde_ = (t - self.t_mean_) / self.t_sd_
        self.n_ = n

        self.stabilizers_ = np.log(
            np.clip(norm.pdf(self.ttilde_), PROBS_MIN, 1.0 - PROBS_MIN)
        )

        beta = fit_linear(self.xtilde_, self.ttilde_)
        sigmasq = float(np.mean((self.ttilde_ - self.xtilde_ @ beta) ** 2))
        self.mle_params_ = np.concatenate([beta, [np.log(sigmasq)]])
        return self

    def transform(self, U: np.ndarray) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        whitened = U[:, self.varying_columns_] @ self.whitener_ - self.whitened_mean_
        return np.column_stack([U[:, self.constant_columns_], whitened])


def _split(params):
    return params[:-1], float(np.exp(params[-1]))


def _log_density(params, design):
    beta, sigmasq = _split(params)
    mean = design.xtilde_ @ beta
    log_density = norm.logpdf(design.ttilde_, loc=mean, scale=np.sqrt(sigmasq))
    return np.clip(log_density, _LOG_MIN, _LOG_MAX)


def _balance_weights(params, design):
    return design.ttilde_ * np.exp(design.stabilizers_ - _log_density(params, design))


def _moments(params, design):
    beta, sigmasq = _split(params)
    x, t, n = design.xtilde_, design.ttilde_, design.n_
    resid = t - x @ beta
    score = x.T @ resid / sigmasq / n
    balance = x.T @ _balance_weights(params, design) / n
    variance = np.array([np.mean(resid**2 / sigmasq - 1.0)])
    return np.concatenate([score, balance, variance])


def _bal_moments(params, design):
    beta, sigmasq = _split(params)
    x, t, n = design.xtilde_, design.ttilde_, design.n_
    resid = t - x @ beta
    balance = x.T @ _balance_weights(params, design) / n
    variance = np.array([np.mean(resid**2 / sigmasq - 1.0)])
    return np.concatenate([balance, variance])


def continuous_weighting_matrix(params, design) -> np.ndarray:
    beta, sigmasq = _split(params)
    x, n = design.xtilde_, design.n_
    k = x.shape[1]
    mean = x @ beta

    v11 = x.T @ x / sigmasq
    v12 = x.T @ x / sigmasq
    v13 = np.zeros((k, 1))
    v22 = x.T @ (x * np.exp(mean**2 / sigmasq + np.log(sigmasq + mean**2))[:, None])
    v23 = (x.T @ mean * 2.0 / sigmasq).reshape(k, 1)
    v33 = np.array([[2.0 * n]])

    v = np.block([[v11, v12, v13], [v12, v22, v23], [v13.T, v23.T, v33]]) / n
    if not np.isfinite(v).all():
        raise ValueError(
            "Encountered an infinite value in the weighting matrix. Use the "
            'just-identified version of CBPS instead by setting method="exact".'
        )
    return v


def continuous_gmm_loss(params, design, inv_v=None) -> float:
    gbar = _moments(params, design)
    if inv_v is None:
        inv_v = pinv_symmetric(continuous_weighting_matrix(params, design))
    return float(gbar @ inv_v @ gbar)


def continuous_bal_loss(params, design) -> float:
    gbar = _bal_moments(params, design)
    return float(gbar @ gbar)


def _jacobians(params, design):
    """Jacobians (moments x params) of the score, balance and variance moments."""
    beta, sigmasq = _split(params)
    x, t, n = design.xtilde_, design.ttilde_, design.n_
    resid = t - x @ beta
    w = _balance_weights(params, design)

    d_score_beta = -x.T @ x / sigmasq / n
    d_score_sig = (-(x * resid[:, None]).sum(axis=0) / sigmasq**2 / n)[:, None]
    d_bal_beta = x.T @ (x * (-resid / sigmasq * w)[:, None]) / n
    d_bal_sig = (
        (x * (w * (1.0 / (2 * sigmasq) - resid**2 / (2 * sigmasq**2)))[:, None]).sum(
            axis=0
        )
        / n
    )[:, None]
    d_var_beta = (x.T @ (-2.0 * resid / sigmasq) / n)[None, :]
    d_var_sig = np.array([[np.sum(-(resid**2) / sigmasq**2) / n]])

    # chain rule: parameter is log sigma^2
    d_score = np.hstack([d_score_beta, d_score_sig * sigmasq])
    d_bal = np.hstack([d_bal_beta, d_bal_sig * sigmasq])
    d_var = np.hstack([d_var_beta, d_var_sig * sigmasq])
    return d_score, d_bal, d_var


def continuous_gmm_gradient(params, design, inv_v) -> np.ndarray:
    gbar = _moments(params, design)
    d_score, d_bal, d_var = _jacobians(params, design)
    return 2.0 * np.vstack([d_score, d_bal, d_var]).T @ inv_v @ gbar


def continuous_bal_gradient(params, design) -> np.ndarray:
    gbar = _bal_moments(params, design)
    _, d_bal, d_var = _jacobians(params, design)
    return 2.0 * np.vstack([d_bal, d_var]).T @ gbar


def fit_cbps_continuous(
    U: np.ndarray,
    t: np.ndarray,
    *,
    method: str = "over",
    twostep: bool = True,
    max_iter: int = 1000,
    standardize: bool = True,
) -> CBPSResult:
    """Fit CBPS for a continuous treatment.

    Returns a :class:`CBPSResult` whose ``beta`` are coefficients on ``U``
    for the conditional mean on the original treatment scale and whose
    ``extra`` holds ``sigmasq`` (original scale), ``beta_tilde``,
    ``sigmasq_tilde``, ``t_mean`` and ``t_sd``.
    """
    U = np.asarray(U, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    bal_only = method == "exact"

    design = ContinuousDesign().fit(U, t)
    mle_params = design.mle_params_
    mle_bal = continuous_bal_loss(mle_params, design)
    try:
        inv_v_init = pinv_symmetric(continuous_weighting_matrix(mle_params, design))
        mle_J = continuous_gmm_loss(mle_params, design)
        params = scale_search(lambda p: continuous_gmm_loss(p, design), mle_params)
    except ValueError as error:
        # R stops here and asks for method="exact"; do that automatically.
        warnings.warn(f"{error} Falling back to the balance-only (exact) objective.")
        bal_only = True
        inv_v_init = None
        mle_J = mle_bal
        params = mle_params

    bal = lambda p: continuous_bal_loss(p, design)  # noqa: E731
    bal_jac = (lambda p: continuous_bal_gradient(p, design)) if twostep else None
    opt_bal = minimize_bfgs(bal, params, jac=bal_jac, max_iter=max_iter)

    if bal_only:
        opt = opt_bal
    else:
        if twostep:
            loss = lambda p: continuous_gmm_loss(p, design, inv_v_init)  # noqa: E731
            jac = lambda p: continuous_gmm_gradient(p, design, inv_v_init)  # noqa: E731
        else:
            loss = lambda p: continuous_gmm_loss(p, design)  # noqa: E731
            jac = None
        opt = pick_best(
            minimize_bfgs(loss, params, jac=jac, max_iter=max_iter),
            minimize_bfgs(loss, opt_bal.x, jac=jac, max_iter=max_iter),
        )

    params_opt = opt.x
    if bal_only:
        J_opt = continuous_bal_loss(params_opt, design)
    else:
        J_opt = (
            continuous_gmm_loss(params_opt, design, inv_v_init)
            if twostep
            else continuous_gmm_loss(params_opt, design)
        )
        if J_opt > mle_J and continuous_bal_loss(params_opt, design) > mle_bal:
            warnings.warn("Optimization failed. Results returned are for MLE.")
            params_opt = mle_params
            J_opt = mle_J

    beta_tilde, sigmasq_tilde = _split(params_opt)
    log_density = _log_density(params_opt, design)
    weights = np.exp(design.stabilizers_ - log_density)
    if standardize:
        weights = weights / weights.sum()

    linear_predictor = design.xtilde_ @ beta_tilde * design.t_sd_ + design.t_mean_
    beta_u = np.linalg.lstsq(U, linear_predictor, rcond=None)[0]
    fitted = np.clip(np.exp(log_density), PROBS_MIN, 1.0 - PROBS_MIN)

    return CBPSResult(
        beta=beta_u,
        fitted_values=fitted,
        weights=weights,
        J=float(J_opt),
        mle_J=float(mle_J),
        converged=bool(opt.success),
        extra={
            "beta_tilde": beta_tilde,
            "sigmasq_tilde": sigmasq_tilde,
            "sigmasq": sigmasq_tilde * design.t_sd_**2,
            "t_mean": design.t_mean_,
            "t_sd": design.t_sd_,
            "bal_loss": continuous_bal_loss(params_opt, design),
            "mle_params": mle_params,
            "inv_v_init": inv_v_init,
            "design": design,
        },
    )
