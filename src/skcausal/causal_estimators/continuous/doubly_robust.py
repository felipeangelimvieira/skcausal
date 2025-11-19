from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.kernel_ridge import KernelRidge

from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor
from typing import Optional
from sklearn.neighbors import KernelDensity

__all__ = ["DoublyRobustPseudoOutcome"]


class DoublyRobustPseudoOutcome(BaseAverageCausalResponseEstimator):
    """
    Doubly-Robust pseudo-outcome for a continuous treatment A.

    Steps:
      1) Fit outcome model μ(x, a) on (X, A) -> Y.
      2) Fit treatment model that outputs sample weights w(X, a).
         We assume w ≈ 1 / π(a|X) (unstabilized) or w ≈ g(a)/π(a|X) (stabilized),
         where g(a) is the marginal density of A.
      3) For each i, compute:
         - π_hat(A_i|X_i) ∝ 1 / w_i(actual)
         - \$_hat(A_i):
             * if weights are stabilized:          \$_hat(A_i) = 1
             * if not stabilized:                  \$_hat(A_i) = g_hat(A_i) via 1D KDE on A
         - μ̄_hat(A_i) = E_X[ μ_hat(X, A_i) ]      (estimated by averaging μ_hat(X, A_i) over X)
         - ξ_i = (Y_i - μ_hat(X_i,A_i)) * [ \$_hat(A_i) / π_hat(A_i|X_i) ] + μ̄_hat(A_i)
      4) Regress ξ_i on A_i with a 1D smoother to get m(a) = E[Y^a].

    The unknown proportional constant in π_hat cancels in the ratio.
    """

    _tags = {
        "capability:predicts_individual": False,
        "capability:supports_multidimensional_treatment": False,
        "t_inner_mtype": pl.DataFrame,
        "store_X": True,
        "one_hot_encode_enum_columns": False,
    }

    def __init__(
        self,
        treatment_regressor: BaseBalancingWeightRegressor,
        outcome_regressor: BaseEstimator,
        pseudo_outcome_regressor: Optional[BaseEstimator] = None,
        mbar_sample_size: Optional[
            int
        ] = 20,  # (still used for μ̄ if you want subsampling later)
        bandwidth: Optional[float] = None,  # for default local-linear smoother
        random_state: int = 0,
    ):
        self.treatment_regressor = treatment_regressor
        self.outcome_regressor = outcome_regressor
        self.pseudo_outcome_regressor = pseudo_outcome_regressor
        self.mbar_sample_size = mbar_sample_size
        self.bandwidth = bandwidth
        self.random_state = random_state
        super().__init__()

    # ---------- helpers ----------
    @staticmethod
    def _as_1d(a):
        return np.asarray(a).ravel()

    @staticmethod
    def _ensure_2d_col(a):
        a = np.asarray(a)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        return a

    def _pi_bar_at(self, a_value: float) -> float:
        """Compute \$_hat(a).

        If weights are stabilized:   return 1.
        Else:                        use KDE of A as estimate of g(a).
        """
        if self.is_stabilized_:
            return 1.0

        # KDE-based marginal density estimate g_hat(a)
        log_d = self.kde_marginal_A_.score_samples(np.array([[a_value]], dtype=float))[
            0
        ]
        return float(np.exp(log_d))

    def _mu_mean_at(self, X: np.ndarray, a_value: float) -> float:
        """Compute μ̄_hat(a) = mean_j μ_hat(X_j, a)."""
        Xt_a = np.concatenate(
            [X, np.full((X.shape[0], 1), a_value, dtype=float)], axis=1
        )
        mu_pred = self.outcome_regressor_.predict(Xt_a)
        return float(self._as_1d(mu_pred).mean())

    # ---------- API ----------
    def _fit(self, X: np.ndarray, y: np.ndarray, t: pl.DataFrame):
        """Fit nuisances (μ, π) and the final smoother g(a) on pseudo-outcomes."""
        self._X = np.asarray(X)
        self._y = self._as_1d(y)
        self._t = t

        if self.treatment_regressor is None:
            raise ValueError(
                "treatment_regressor must be provided for DR pseudo-outcome."
            )

        # 1) Fit treatment model
        self.treatment_regressor_ = self.treatment_regressor.clone()
        self.treatment_regressor_.fit(self._X, t)

        self.is_stabilized_ = (
            self.treatment_regressor_.get_tag("balancing_weight_type") == "stabilized"
        )

        # 2) Fit outcome model μ(x, a)
        t_vec = self._as_1d(t.to_numpy())
        Xt = np.concatenate([self._X, t_vec.reshape(-1, 1)], axis=1)
        self.outcome_regressor_ = deepcopy(self.outcome_regressor)
        self.outcome_regressor_.fit(Xt, self._y)

        eps = 1e-8

        # 3) π_hat(A_i|X_i) from actual weights
        w_actual = self.treatment_regressor_.predict_sample_weight(self._X, t)
        w_actual = self._as_1d(w_actual)

        # For both stabilized and unstabilized, π_hat is proportional to 1 / w_actual
        pi_actual = 1.0 / np.clip(w_actual, eps, None)

        # 4) If NOT stabilized: fit KDE on A to estimate g(a)
        if not self.is_stabilized_:
            t_vals = t_vec.reshape(-1, 1)
            t_std = np.std(t_vec)
            if t_std <= 0:
                t_std = 1.0
            bw = 1.06 * t_std * (len(t_vec) ** (-1.0 / 5.0))
            self.kde_marginal_A_ = KernelDensity(kernel="gaussian", bandwidth=bw).fit(
                t_vals
            )
        else:
            self.kde_marginal_A_ = None

        # 5) μ_hat(X_i, A_i)
        mu_actual = self._as_1d(self.outcome_regressor_.predict(Xt))

        # 6) Compute \$_hat(A_i) and μ̄_hat(A_i) for each i
        #    (this is the O(n^2) part via μ̄; can be optimized with grids if needed)
        pi_bar = np.array(
            [self._pi_bar_at(a) for a in t_vec],
            dtype=float,
        )
        mu_bar = np.array(
            [self._mu_mean_at(self._X, a) for a in t_vec],
            dtype=float,
        )

        # 7) Pseudo-outcome: ξ_i = (Y - μ(X,A)) * (\$ / π) + μ̄(A)
        ratio = pi_bar / np.clip(pi_actual, eps, None)
        xi = (self._y - mu_actual) * ratio + mu_bar

        self.xi_ = xi
        self.a_train_ = t_vec

        # 8) Final smoother g(a)
        self.pseudo_outcome_regressor_ = deepcopy(self.pseudo_outcome_regressor)

        self.pseudo_outcome_regressor_.fit(self.a_train_.reshape(-1, 1), self.xi_)

        return self

    def _predict_adrf(self, X: np.ndarray, t: pl.DataFrame) -> np.ndarray:
        """Predict the average dose-response m(a)=E[Y^a] at the provided treatment grid."""
        t_vec = self._as_1d(t.to_numpy())
        return self.pseudo_outcome_regressor_.predict(t_vec.reshape(-1, 1))

    def _predict_average_treatment_effect(
        self, X: np.ndarray, t: pl.DataFrame
    ) -> float:
        """Return the mean of the ADRF values over the provided grid t."""
        vals = self._predict_adrf(X, t)
        return float(np.mean(vals))
