"""Maximum-likelihood initialisers used as CBPS starting values.

R's CBPS starts the GMM optimisation at the GLM / multinomial / OLS maximum
likelihood estimates. These helpers reproduce those fits with numpy/scipy so
no additional dependency is needed.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["fit_logistic", "fit_multinomial", "fit_linear", "softmax_probs"]


def softmax_probs(U: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Multinomial-logit probabilities with level 0 as the baseline.

    Parameters
    ----------
    U : ndarray of shape (n, k)
    beta : ndarray of shape (k, J - 1)

    Returns
    -------
    ndarray of shape (n, J)
    """
    theta = U @ beta
    logits = np.column_stack([np.zeros(theta.shape[0]), theta])
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def fit_multinomial(U: np.ndarray, T: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Unpenalised multinomial logit MLE (baseline = first column of ``T``).

    Returns coefficients of shape ``(k, J - 1)``.
    """
    U = np.asarray(U, dtype=float)
    T = np.asarray(T, dtype=float)
    n, k = U.shape
    J = T.shape[1]

    def negloglik_and_grad(flat):
        beta = flat.reshape(k, J - 1)
        probs = softmax_probs(U, beta)
        loglik = np.sum(T * np.log(np.clip(probs, 1e-300, None)))
        grad = U.T @ (probs[:, 1:] - T[:, 1:])
        return -loglik, grad.ravel()

    result = minimize(
        negloglik_and_grad,
        np.zeros(k * (J - 1)),
        jac=True,
        method="BFGS",
        options={"gtol": tol, "maxiter": 10_000},
    )
    return result.x.reshape(k, J - 1)


def fit_logistic(U: np.ndarray, y: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Unpenalised logistic-regression MLE; returns coefficients of shape ``(k,)``."""
    y = np.asarray(y, dtype=float).ravel()
    T = np.column_stack([1.0 - y, y])
    return fit_multinomial(U, T, tol=tol)[:, 0]


def fit_linear(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary least squares; returns coefficients of shape ``(k,)``."""
    y = np.asarray(y, dtype=float).ravel()
    return np.linalg.lstsq(np.asarray(U, dtype=float), y, rcond=None)[0]
