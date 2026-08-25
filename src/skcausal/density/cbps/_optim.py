"""Optimisation helpers shared by the CBPS fitters.

These reproduce the control flow of the R implementation: a one-dimensional
rescaling of the MLE start on ``[0.8, 1.1]``, BFGS runs from both the MLE
start and the balance-only optimum, and selection of the better run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar

__all__ = ["scale_search", "minimize_bfgs", "OptimResult", "pick_best", "pinv_symmetric"]

PROBS_MIN = 1e-6


def pinv_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Moore-Penrose inverse of a symmetric matrix, symmetrised.

    The GMM weighting matrices can be badly conditioned; ``np.linalg.pinv``
    then returns a noticeably asymmetric result, which breaks the gradient
    identity ``d(g'Ag)/dp = 2 J'Ag``. Averaging with the transpose restores
    symmetry.
    """
    inverse = np.linalg.pinv(matrix, hermitian=True)
    return 0.5 * (inverse + inverse.T)


@dataclass
class OptimResult:
    x: np.ndarray
    fun: float
    success: bool


def scale_search(loss: Callable[[np.ndarray], float], params: np.ndarray) -> np.ndarray:
    """Rescale ``params`` by the scalar in ``[0.8, 1.1]`` minimising ``loss``."""
    result = minimize_scalar(
        lambda alpha: float(loss(params * alpha)),
        bounds=(0.8, 1.1),
        method="bounded",
        options={"xatol": np.finfo(float).eps ** 0.25},
    )
    return params * result.x


def minimize_bfgs(
    loss: Callable[[np.ndarray], float],
    x0: np.ndarray,
    jac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_iter: int = 1000,
) -> OptimResult:
    """BFGS with tight tolerances; falls back to Nelder-Mead on failure."""
    x0 = np.asarray(x0, dtype=float)
    try:
        result = minimize(
            loss,
            x0,
            jac=jac,
            method="BFGS",
            options={"gtol": 1e-10, "maxiter": max_iter},
        )
        x, fun, success = result.x, float(result.fun), bool(result.success)
        if not np.isfinite(fun):
            raise FloatingPointError("non-finite objective")
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        result = minimize(
            loss, x0, method="Nelder-Mead", options={"maxiter": max_iter}
        )
        x, fun, success = result.x, float(result.fun), bool(result.success)

    # scipy may report failure on precision loss while sitting at the optimum;
    # treat "no worse than the start and finite" as converged for that case.
    if not success and np.isfinite(fun) and fun <= float(loss(x0)) + 1e-12:
        success = True
    return OptimResult(x=x, fun=fun, success=success)


def pick_best(first: OptimResult, second: OptimResult) -> OptimResult:
    """Return the run with the smaller objective (ties go to ``second``, as R)."""
    return first if first.fun < second.fun else second
