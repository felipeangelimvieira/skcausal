"""Design-matrix preprocessing shared by all CBPS routines.

Mirrors the preprocessing in R's ``CBPS.fit``: an intercept is prepended,
the remaining columns are z-scored (``ddof=1``), and the augmented matrix is
replaced by the left singular vectors of its SVD. The CBPS fitters then work
with an orthonormal design, which makes ``X'X`` the identity.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DesignStandardizer"]


class DesignStandardizer:
    """Intercept + z-score + SVD transform with coefficient back-mapping.

    Attributes
    ----------
    mean_ : ndarray of shape (p,)
        Column means of the raw covariates.
    sd_ : ndarray of shape (p,)
        Column standard deviations (``ddof=1``) of the raw covariates.
    u_, d_, v_ : ndarray
        Thin SVD of the augmented, standardized design ``[1, z(X)] = U diag(d) V'``.
    k : int
        Rank of the design (equals ``p + 1`` when it is full rank).
    """

    def fit(self, X: np.ndarray) -> "DesignStandardizer":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        self.mean_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros(X.shape[1])
        constant = np.flatnonzero(self.sd_ <= 1e-10)
        if constant.size:
            raise ValueError(
                "Covariate columns at positions "
                f"{constant.tolist()} are constant. CBPS adds an intercept "
                "internally; remove constant columns from X."
            )

        augmented = self._augment(X)
        u, d, vt = np.linalg.svd(augmented, full_matrices=False)
        rank = int(np.sum(d > d[0] * max(augmented.shape) * np.finfo(float).eps))
        if rank < augmented.shape[1]:
            raise ValueError("X is not full rank.")

        self.u_ = u
        self.d_ = d
        self.v_ = vt.T
        self.k = augmented.shape[1]
        return self

    def _augment(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = (X - self.mean_) / self.sd_
        return np.column_stack([np.ones(X.shape[0]), z])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map raw covariates into the orthonormal ``U`` space."""
        return self._augment(X) @ self.v_ / self.d_

    def coef_to_original(self, beta_u: np.ndarray) -> np.ndarray:
        """Map coefficients on ``U`` to coefficients on ``[1, X]``.

        Follows the reversal in R's ``CBPS.fit``: undo the SVD, divide slopes
        by the column standard deviations and shift the intercept.
        """
        beta_u = np.asarray(beta_u, dtype=float)
        squeeze = beta_u.ndim == 1
        beta_u = beta_u.reshape(beta_u.shape[0], -1)

        beta = self.v_ @ (beta_u / self.d_[:, None])
        beta[1:] = beta[1:] / self.sd_[:, None]
        beta[0] = beta[0] - self.mean_ @ beta[1:]
        return beta[:, 0] if squeeze else beta
