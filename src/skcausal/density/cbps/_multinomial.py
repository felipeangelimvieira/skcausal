"""CBPS for categorical treatments with ``J >= 2`` levels (ATE weights).

This generalises R's ``CBPS.2Treat`` (ATE), ``CBPS.3Treat`` and
``CBPS.4Treat``. The moment conditions are, per unit ``i`` with design row
``u_i`` and one-hot treatment ``T_i``:

* score moments ``u_i (T_ij - pi_ij)`` for ``j = 1..J-1``;
* balance moments ``u_i (T_ij / pi_ij - T_i0 / pi_i0)`` for ``j = 1..J-1``.

R uses different (but equivalent) contrasts for 3 and 4 levels. Because the
GMM weighting matrix is the model-implied covariance of the moments, the
"over" objective is invariant to the contrast basis, and the "exact"
solution (all balance moments zero) is basis-independent too.

Parameters are stored as ``beta`` of shape ``(k, J - 1)`` and flattened in
column-major order (one block of ``k`` coefficients per non-baseline level),
matching R's ``matrix(beta, k, J - 1)``.
"""

from __future__ import annotations

import warnings

import numpy as np

from skcausal.density.cbps._glm import fit_multinomial, softmax_probs
from skcausal.density.cbps._optim import (
    PROBS_MIN,
    minimize_bfgs,
    pick_best,
    pinv_symmetric,
    scale_search,
)
from skcausal.density.cbps._result import CBPSResult

__all__ = [
    "multinomial_probs",
    "gmm_moments",
    "gmm_weighting_matrix",
    "gmm_loss",
    "gmm_gradient",
    "bal_loss",
    "bal_gradient",
    "fit_cbps_multinomial",
]


def _unflatten(params: np.ndarray, k: int) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    return params.reshape(k, -1, order="F")


def multinomial_probs(U: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Clipped and renormalised multinomial-logit probabilities ``(n, J)``."""
    beta = _unflatten(params, U.shape[1])
    probs = np.maximum(softmax_probs(U, beta), PROBS_MIN)
    return probs / probs.sum(axis=1, keepdims=True)


def _balance_contrasts(T: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """``T_j / pi_j - T_0 / pi_0`` for ``j >= 1``; shape ``(n, J - 1)``."""
    ratio = T / probs
    return ratio[:, 1:] - ratio[:, [0]]


def gmm_moments(U: np.ndarray, T: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Stacked sample moments ``g_bar`` of length ``2 k (J - 1)``."""
    n = U.shape[0]
    score = U.T @ (T[:, 1:] - probs[:, 1:]) / n
    balance = U.T @ _balance_contrasts(T, probs) / n
    return np.concatenate([score.ravel(order="F"), balance.ravel(order="F")])


def _bal_moments(U: np.ndarray, T: np.ndarray, probs: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    return (U.T @ _balance_contrasts(T, probs) / n).ravel(order="F")


def gmm_weighting_matrix(U: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Model-implied covariance ``V`` of the moment conditions."""
    n, k = U.shape
    J = probs.shape[1]
    m = J - 1
    blocks = [[None] * (2 * m) for _ in range(2 * m)]

    def block(weight):
        return U.T @ (U * weight[:, None]) / n

    for a in range(m):
        for b in range(m):
            pa, pb = probs[:, a + 1], probs[:, b + 1]
            score_score = pa * (1.0 - pa) if a == b else -pa * pb
            blocks[a][b] = block(score_score)
            blocks[a][m + b] = block(np.full(n, 1.0 if a == b else 0.0))
            blocks[m + a][b] = blocks[a][m + b]
            bal_bal = 1.0 / probs[:, 0] + (1.0 / pa if a == b else 0.0)
            blocks[m + a][m + b] = block(bal_bal)
    return np.block(blocks)


def gmm_loss(params, U, T, inv_v=None) -> float:
    probs = multinomial_probs(U, params)
    gbar = gmm_moments(U, T, probs)
    if inv_v is None:
        inv_v = pinv_symmetric(gmm_weighting_matrix(U, probs))
    return float(gbar @ inv_v @ gbar)


def bal_loss(params, U, T) -> float:
    gbar = _bal_moments(U, T, multinomial_probs(U, params))
    return float(gbar @ gbar)


def _moment_jacobians(U, T, probs):
    """Jacobians of score and balance moments w.r.t. the flattened params.

    Returns ``(d_score, d_bal)`` each of shape ``(k (J - 1), k (J - 1))``
    with rows indexed by moments and columns by parameters.
    """
    n, k = U.shape
    m = probs.shape[1] - 1
    d_score = np.zeros((k * m, k * m))
    d_bal = np.zeros((k * m, k * m))
    ratio = T / probs
    for a in range(m):  # moment level a + 1
        pa = probs[:, a + 1]
        for l in range(m):  # parameter level l + 1
            pl = probs[:, l + 1]
            delta = 1.0 if a == l else 0.0
            d_score_w = -pa * (delta - pl)
            d_bal_w = -ratio[:, a + 1] * delta + ratio[:, a + 1] * pl - ratio[:, 0] * pl
            rows, cols = slice(a * k, (a + 1) * k), slice(l * k, (l + 1) * k)
            d_score[rows, cols] = U.T @ (U * d_score_w[:, None]) / n
            d_bal[rows, cols] = U.T @ (U * d_bal_w[:, None]) / n
    return d_score, d_bal


def gmm_gradient(params, U, T, inv_v) -> np.ndarray:
    probs = multinomial_probs(U, params)
    gbar = gmm_moments(U, T, probs)
    d_score, d_bal = _moment_jacobians(U, T, probs)
    d_gbar = np.vstack([d_score, d_bal])
    return 2.0 * d_gbar.T @ inv_v @ gbar


def bal_gradient(params, U, T) -> np.ndarray:
    probs = multinomial_probs(U, params)
    gbar = _bal_moments(U, T, probs)
    _, d_bal = _moment_jacobians(U, T, probs)
    return 2.0 * d_bal.T @ gbar


def fit_cbps_multinomial(
    U: np.ndarray,
    T: np.ndarray,
    *,
    method: str = "over",
    twostep: bool = True,
    max_iter: int = 1000,
    standardize: bool = True,
    mle_fallback: bool = True,
) -> CBPSResult:
    """Fit CBPS for a categorical treatment (ATE weights).

    Parameters
    ----------
    U : ndarray of shape (n, k)
        Orthonormal design (see ``DesignStandardizer``).
    T : ndarray of shape (n, J)
        One-hot treatment indicators; column 0 is the baseline level.
    method : {"over", "exact"}
    twostep : bool
        Fixed weighting matrix (from the rescaled MLE) with analytic
        gradients, or continuous-updating GMM.
    max_iter : int
    standardize : bool
        Normalise weights to sum to one within each level.
    mle_fallback : bool
        Return the MLE when the optimum is worse than the MLE on both the
        GMM and balance objectives (R does this for 3+ levels only).
    """
    U = np.asarray(U, dtype=float)
    T = np.asarray(T, dtype=float)
    n, k = U.shape
    J = T.shape[1]
    bal_only = method == "exact"

    mle_beta = fit_multinomial(U, T)
    mle_params = mle_beta.ravel(order="F")

    params = scale_search(lambda p: gmm_loss(p, U, T), mle_params)
    inv_v_init = pinv_symmetric(gmm_weighting_matrix(U, multinomial_probs(U, params)))

    if twostep:
        opt_bal = minimize_bfgs(
            lambda p: bal_loss(p, U, T),
            params,
            jac=lambda p: bal_gradient(p, U, T),
            max_iter=max_iter,
        )
    else:
        opt_bal = minimize_bfgs(lambda p: bal_loss(p, U, T), params, max_iter=max_iter)

    if bal_only:
        opt = opt_bal
    else:
        if twostep:
            loss = lambda p: gmm_loss(p, U, T, inv_v_init)  # noqa: E731
            jac = lambda p: gmm_gradient(p, U, T, inv_v_init)  # noqa: E731
        else:
            loss = lambda p: gmm_loss(p, U, T)  # noqa: E731
            jac = None
        opt_glm_init = minimize_bfgs(loss, params, jac=jac, max_iter=max_iter)
        opt_bal_init = minimize_bfgs(loss, opt_bal.x, jac=jac, max_iter=max_iter)
        opt = pick_best(opt_glm_init, opt_bal_init)

    params_opt = opt.x
    probs_opt = multinomial_probs(U, params_opt)
    converged = opt.success

    if twostep:
        J_opt = gmm_loss(params_opt, U, T, inv_v_init)
        mle_J = gmm_loss(mle_params, U, T, inv_v_init)
    else:
        J_opt = gmm_loss(params_opt, U, T)
        mle_J = gmm_loss(mle_params, U, T)

    if (
        mle_fallback
        and J > 2
        and not bal_only
        and J_opt > gmm_loss(mle_params, U, T)
        and bal_loss(params_opt, U, T) > bal_loss(mle_params, U, T)
    ):
        warnings.warn("Optimization failed. Results returned are for MLE.")
        params_opt = mle_params
        probs_opt = multinomial_probs(U, params_opt)
        J_opt = gmm_loss(mle_params, U, T)

    inverse = T / probs_opt
    norms = inverse.sum(axis=0) if standardize else np.ones(J)
    weights = (inverse / norms).sum(axis=1)

    return CBPSResult(
        beta=_unflatten(params_opt, k),
        fitted_values=probs_opt,
        weights=weights,
        J=float(J_opt),
        mle_J=float(mle_J),
        converged=bool(converged),
        extra={"bal_loss": bal_loss(params_opt, U, T), "mle_beta": mle_beta, "inv_v_init": inv_v_init},
    )
