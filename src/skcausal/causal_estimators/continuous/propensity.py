from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.kernel_ridge import KernelRidge

from skcausal.causal_estimators.base import BaseCausalResponseEstimator, to_dummies
from skcausal.utils.polars import convert_categorical_to_dummies
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor
from sklearn.neighbors import KernelDensity


__all__ = [
    "PropensityWeightingContinuous",
    "PropensityPseudoOutcomeContinuous",
]


class PropensityWeightingContinuous(BaseCausalResponseEstimator):
    """
    Uses Propensity Score Weighting to forecast the average treatment effect of Discrete Treatments.

    Parameters
    ----------
    treatment_regressor : BaseSampleWeightRegressor
        Regressor to estimate the propensity score.
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
        kernel: Optional[Union[Kernel, KernelDensity]] = None,
        self_normalized: bool = False,
        random_state=0,
    ):
        self.treatment_regressor = treatment_regressor
        self.kernel = kernel
        self.self_normalized = self_normalized
        self.random_state = random_state

        super().__init__()

        self._outcome_regressor = kernel
        self._kernel = RBF() if kernel is None else kernel

    def _fit(self, X: np.ndarray, y: np.ndarray, t: pl.DataFrame):
        """Fits the GPS estimator.


        First, fits the treatment regressor to estimate the propensity score.
        Then, fits the outcome regressor to estimate the outcome.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target variable.
        t : np.ndarray

        Returns
        -------
        self
            The object itself
        """

        self._X = X
        self._y = y
        self._t = t

        self._rng = np.random.default_rng(self.random_state)

        if self.treatment_regressor is not None:
            self.treatment_regressor_ = self.treatment_regressor.clone()
            self.treatment_regressor_.fit(X, t)
            self._w = self.treatment_regressor_.predict_sample_weight(X, t).reshape(-1)
        else:
            self.treatment_regressor_ = None
            self._w = np.ones(X.shape[0])

        self._kernel = self._resolve_kernel()

        if isinstance(self._kernel, KernelDensity):
            # Fit a fresh KernelDensity instance on observed treatments
            self._kernel.fit(t.to_numpy().reshape(-1, 1))

        return self

    def _predict_average_treatment_effect(self, X, t) -> float:
        """Predict the average treatment effect for the given treatment values t.

        Parameters
        ----------
        X : np.ndarray
            The input data
        t : np.ndarray
            The treatment values

        Returns
        -------
        float
            The average treatment effect for the given treatment values t.
        """

        return np.array(self._predict_adrf(X, t)).reshape((-1, 1)).mean()

    def _predict_adrf(self, X: np.ndarray, t: pl.DataFrame) -> list[float]:
        """
        Predict the average response for each treatment value in t.

        Parameters
        ----------
        X : np.ndarray
            The input data
        t : list[float]
            The treatment values. Ignored since for binary treatments the values can only be False and True

        Returns
        -------
        list[float]
            The average response for each treatment value in t.
        """

        # We ignore X at prediction-time; this is a marginal dose-response estimator.
        t_grid = _to_column_vector(t.to_numpy())  # shape (m, 1)
        t_obs = _to_column_vector(self._t.to_numpy())  # shape (n, 1)
        y = self._y.reshape(-1)
        w = self._w.reshape(-1)

        # Kernel between observed T and target grid
        K = _evaluate_kernel_matrix(self._kernel, t_obs, t_grid)  # (n, m)
        K = K / K.sum(axis=0, keepdims=True)  # normalize per column
        num = (K * (w[:, None] * y[:, None])).sum(axis=0)  # length m
        is_stabilized = self.treatment_regressor_.get_tag("balancing_weight_type")
        if self.self_normalized:
            den = (K * w[:, None]).sum(axis=0)
        elif is_stabilized:
            den = 1
        elif not is_stabilized:
            den = self._X.shape[0]
        # n_samples = self._X.shape[0]
        # den = n_samples

        # Avoid division by zero; could add eps
        mu = num / den

        return mu.reshape(-1, 1)

    def _resolve_kernel(
        self,
    ) -> Union[Kernel, KernelDensity, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        kernel = self.kernel

        if kernel is None:
            return RBF()

        if isinstance(kernel, (Kernel, KernelDensity)):
            return deepcopy(kernel)

        if callable(kernel):
            return kernel

        raise TypeError(
            "`kernel` must be an sklearn kernel, KernelDensity, or callable returning a kernel matrix."
        )


def _to_column_vector(x):
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _evaluate_kernel_matrix(
    kernel: Union[
        Kernel, KernelDensity, Callable[[np.ndarray, np.ndarray], np.ndarray]
    ],
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    if isinstance(kernel, KernelDensity):
        return _kernel_density_matrix(kernel, x, y)

    if isinstance(kernel, Kernel):
        return kernel(x, y)

    if callable(kernel):
        return kernel(x, y)

    raise TypeError(
        "`kernel` must be an instance of sklearn.gaussian_process.kernels.Kernel, "
        "sklearn.neighbors.KernelDensity, or a callable returning a kernel matrix."
    )


def _kernel_density_matrix(
    kde: KernelDensity, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    if kde.kernel != "gaussian":
        raise ValueError(
            "KernelDensity kernels other than 'gaussian' are not supported for "
            "pairwise evaluation."
        )

    bandwidth = float(kde.bandwidth)
    if bandwidth <= 0:
        raise ValueError("KernelDensity bandwidth must be positive.")

    # Gaussian kernel without normalization constant; it cancels in the ADRF ratio.
    diff = (x - y.T) / bandwidth
    return np.exp(-0.5 * diff**2)


class PropensityPseudoOutcomeContinuous(BaseCausalResponseEstimator):
    """Pseudo-outcome estimator that relies solely on propensity score weights.

    Steps:
      1) Fit treatment model that outputs sample weights ``w(X, a)``.
         We assume ``w ≈ 1 / π(a|X)`` (unstabilized) or ``w ≈ g(a) / π(a|X)`` (stabilized),
         where ``g(a)`` is the marginal density of ``A``.
      2) For each observation ``i`` compute:
         - ``π_hat(A_i|X_i) ∝ 1 / w_i``
         - ``π̄_hat(A_i)``:
              * if weights are stabilized:          ``π̄_hat(A_i) = 1``
              * if not stabilized:                  ``π̄_hat(A_i) = g_hat(A_i)`` via 1D KDE on ``A``
         - ``ξ_i = Y_i * [π̄_hat(A_i) / π_hat(A_i|X_i)]``
      3) Regress the pseudo outcomes ``ξ_i`` on ``A_i`` with a 1D smoother to recover the ADRF.
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
        pseudo_outcome_regressor: BaseEstimator,
        bandwidth: Optional[float] = None,
        random_state: int = 0,
    ):
        if pseudo_outcome_regressor is None:
            raise ValueError(
                "pseudo_outcome_regressor must be provided for propensity pseudo-outcomes."
            )

        self.treatment_regressor = treatment_regressor
        self.pseudo_outcome_regressor = pseudo_outcome_regressor
        self.bandwidth = bandwidth
        self.random_state = random_state

        super().__init__()

    @staticmethod
    def _as_1d(a: np.ndarray) -> np.ndarray:
        return np.asarray(a).ravel()

    def _pi_bar_at(self, a_value: float) -> float:
        if self.is_stabilized_:
            return 1.0

        log_density = self.kde_marginal_A_.score_samples(
            np.array([[a_value]], dtype=float)
        )[0]
        return float(np.exp(log_density))

    def _fit(self, X: np.ndarray, y: np.ndarray, t: pl.DataFrame):
        self._X = np.asarray(X)
        self._y = self._as_1d(y)
        self._t = t

        if self.treatment_regressor is None:
            raise ValueError(
                "treatment_regressor must be provided for propensity pseudo-outcome."
            )

        self.treatment_regressor_ = self.treatment_regressor.clone()
        self.treatment_regressor_.fit(self._X, t)

        self.is_stabilized_ = (
            self.treatment_regressor_.get_tag("balancing_weight_type") == "stabilized"
        )

        w_actual = self._as_1d(
            self.treatment_regressor_.predict_sample_weight(self._X, t)
        )

        eps = 1e-8
        pi_actual = 1.0 / np.clip(w_actual, eps, None)

        t_vec = self._as_1d(t.to_numpy())

        if not self.is_stabilized_:
            t_vals = t_vec.reshape(-1, 1)
            if self.bandwidth is not None:
                bw = float(self.bandwidth)
            else:
                t_std = np.std(t_vec)
                if t_std <= 0:
                    t_std = 1.0
                bw = 1.06 * t_std * (len(t_vec) ** (-1.0 / 5.0))
            self.kde_marginal_A_ = KernelDensity(kernel="gaussian", bandwidth=bw).fit(
                t_vals
            )
            self.bandwidth_ = bw
        else:
            self.kde_marginal_A_ = None
            self.bandwidth_ = None

        pi_bar = np.array([self._pi_bar_at(a) for a in t_vec], dtype=float)

        ratio = pi_bar / np.clip(pi_actual, eps, None)
        xi = self._y * ratio

        self.xi_ = xi
        self.a_train_ = t_vec

        self.pseudo_outcome_regressor_ = deepcopy(self.pseudo_outcome_regressor)
        self.pseudo_outcome_regressor_.fit(self.a_train_.reshape(-1, 1), self.xi_)

        return self

    def _predict_adrf(self, X: np.ndarray, t: pl.DataFrame) -> np.ndarray:
        t_vec = self._as_1d(t.to_numpy())
        return self.pseudo_outcome_regressor_.predict(t_vec.reshape(-1, 1))

    def _predict_average_treatment_effect(
        self, X: np.ndarray, t: pl.DataFrame
    ) -> float:
        vals = self._predict_adrf(X, t)
        return float(np.mean(vals))
