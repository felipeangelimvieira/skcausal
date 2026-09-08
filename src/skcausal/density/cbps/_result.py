"""Container for the output of the CBPS core fitters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["CBPSResult"]


@dataclass
class CBPSResult:
    """Result of a CBPS fit in the orthonormal design space.

    Attributes
    ----------
    beta : ndarray
        Coefficients on the ``U`` design (``(k,)`` or ``(k, J - 1)``).
    fitted_values : ndarray
        Fitted probabilities ``(n, J)`` / ``(n,)`` or Gaussian densities.
    weights : ndarray of shape (n,)
        Balancing weights, standardised as requested.
    J : float
        Objective value at the optimum (GMM loss for ``method="over"``,
        balance loss for ``method="exact"`` in the continuous case, GMM loss
        otherwise, mirroring R).
    mle_J : float
        Objective value at the maximum-likelihood starting values.
    converged : bool
        Whether the selected optimisation run reported convergence.
    extra : dict
        Routine-specific extras (e.g. ``sigmasq_tilde`` for continuous).
    """

    beta: np.ndarray
    fitted_values: np.ndarray
    weights: np.ndarray
    J: float
    mle_J: float
    converged: bool
    extra: dict[str, Any] = field(default_factory=dict)
