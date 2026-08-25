"""CBPS for a binary treatment targeting the treated population (ATT).

Mirrors the ``ATT`` branch of R's ``CBPS.2Treat``. With ``pi_i`` the
propensity score, ``n`` the sample size and ``n_t`` the number of treated
units, the balancing weight is ``w_i = n / n_t * (T_i - pi_i) / (1 - pi_i)``
(equal to ``n / n_t`` for treated units and ``-n / n_t * pi_i / (1 - pi_i)``
for controls) and the moment conditions are ``u_i (T_i - pi_i)`` and
``u_i w_i``.
"""

from __future__ import annotations

import numpy as np

from skcausal.density.cbps._glm import fit_logistic
from skcausal.density.cbps._optim import (
    PROBS_MIN,
    minimize_bfgs,
    pick_best,
    pinv_symmetric,
    scale_search,
)
from skcausal.density.cbps._result import CBPSResult

__all__ = [
    "att_probs",
    "att_weights",
    "att_weighting_matrix",
    "att_gmm_loss",
    "att_gmm_gradient",
    "att_bal_loss",
    "att_bal_gradient",
    "fit_cbps_binary_att",
]


def att_probs(U: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Clipped logistic probabilities ``P(T = 1 | u)``."""
    probs = 1.0 / (1.0 + np.exp(-(U @ params)))
    return np.clip(probs, PROBS_MIN, 1.0 - PROBS_MIN)


def att_weights(treat: np.ndarray, probs: np.ndarray) -> np.ndarray:
    n, n_t = len(treat), treat.sum()
    return n / n_t * (treat - probs) / (1.0 - probs)


def _moments(U, treat, probs):
    n = U.shape[0]
    score = U.T @ (treat - probs) / n
    balance = U.T @ att_weights(treat, probs) / n
    return np.concatenate([score, balance])


def _bal_moments(U, treat, probs):
    return U.T @ att_weights(treat, probs) / U.shape[0]


def att_weighting_matrix(U, treat, probs) -> np.ndarray:
    n, n_t = len(treat), treat.sum()
    ratio = n / n_t

    def block(weight):
        return U.T @ (U * weight[:, None]) / n

    v11 = block(probs * (1.0 - probs))
    v12 = block(probs) * ratio
    v22 = block(probs / (1.0 - probs)) * ratio**2
    return np.block([[v11, v12], [v12, v22]])


def att_gmm_loss(params, U, treat, inv_v=None) -> float:
    probs = att_probs(U, params)
    gbar = _moments(U, treat, probs)
    if inv_v is None:
        inv_v = pinv_symmetric(att_weighting_matrix(U, treat, probs))
    return float(gbar @ inv_v @ gbar)


def att_bal_loss(params, U, treat) -> float:
    gbar = _bal_moments(U, treat, att_probs(U, params))
    return float(gbar @ gbar)


def _jacobians(U, treat, probs):
    n, n_t = len(treat), treat.sum()
    d_score = -U.T @ (U * (probs * (1.0 - probs))[:, None]) / n
    dw = np.where(treat == 1, 0.0, -n / n_t * probs / (1.0 - probs))
    d_bal = U.T @ (U * dw[:, None]) / n
    return d_score, d_bal


def att_gmm_gradient(params, U, treat, inv_v) -> np.ndarray:
    probs = att_probs(U, params)
    gbar = _moments(U, treat, probs)
    d_score, d_bal = _jacobians(U, treat, probs)
    return 2.0 * np.vstack([d_score, d_bal]).T @ inv_v @ gbar


def att_bal_gradient(params, U, treat) -> np.ndarray:
    probs = att_probs(U, params)
    gbar = _bal_moments(U, treat, probs)
    _, d_bal = _jacobians(U, treat, probs)
    return 2.0 * d_bal.T @ gbar


def fit_cbps_binary_att(
    U: np.ndarray,
    treat: np.ndarray,
    *,
    method: str = "over",
    twostep: bool = True,
    max_iter: int = 1000,
    standardize: bool = True,
) -> CBPSResult:
    """Fit binary CBPS with ATT weights (treated coded as ``treat == 1``)."""
    U = np.asarray(U, dtype=float)
    treat = np.asarray(treat, dtype=float).ravel()
    bal_only = method == "exact"

    mle_params = fit_logistic(U, treat)
    params = scale_search(lambda p: att_gmm_loss(p, U, treat), mle_params)
    inv_v_init = pinv_symmetric(att_weighting_matrix(U, treat, att_probs(U, params)))

    bal = lambda p: att_bal_loss(p, U, treat)  # noqa: E731
    bal_jac = (lambda p: att_bal_gradient(p, U, treat)) if twostep else None
    opt_bal = minimize_bfgs(bal, params, jac=bal_jac, max_iter=max_iter)

    if bal_only:
        opt = opt_bal
    else:
        if twostep:
            loss = lambda p: att_gmm_loss(p, U, treat, inv_v_init)  # noqa: E731
            jac = lambda p: att_gmm_gradient(p, U, treat, inv_v_init)  # noqa: E731
        else:
            loss = lambda p: att_gmm_loss(p, U, treat)  # noqa: E731
            jac = None
        opt = pick_best(
            minimize_bfgs(loss, params, jac=jac, max_iter=max_iter),
            minimize_bfgs(loss, opt_bal.x, jac=jac, max_iter=max_iter),
        )

    params_opt = opt.x
    probs_opt = att_probs(U, params_opt)
    if twostep:
        J_opt = att_gmm_loss(params_opt, U, treat, inv_v_init)
        mle_J = att_gmm_loss(mle_params, U, treat, inv_v_init)
    else:
        J_opt = att_gmm_loss(params_opt, U, treat)
        mle_J = att_gmm_loss(mle_params, U, treat)

    n, n_t = len(treat), treat.sum()
    treated_w = np.where(treat == 1, n / n_t, 0.0)
    control_w = np.where(treat == 0, n / n_t * probs_opt / (1.0 - probs_opt), 0.0)
    if standardize:
        treated_w = treated_w / treated_w.sum()
        control_w = control_w / control_w.sum()
    weights = treated_w + control_w

    return CBPSResult(
        beta=params_opt,
        fitted_values=probs_opt,
        weights=weights,
        J=float(J_opt),
        mle_J=float(mle_J),
        converged=bool(opt.success),
        extra={"bal_loss": att_bal_loss(params_opt, U, treat), "mle_beta": mle_params, "inv_v_init": inv_v_init},
    )
