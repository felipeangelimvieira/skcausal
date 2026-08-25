import numpy as np
import pandas as pd
from scipy.stats import norm

from skcausal.density.base import BaseDensityEstimator
from skcausal.density.cbps._common import covariates_to_array, validate_method
from skcausal.density.cbps._continuous import fit_cbps_continuous
from skcausal.density.cbps._design import DesignStandardizer

__all__ = ["CBPSContinuous"]


class CBPSContinuous(BaseDensityEstimator):
    """Covariate Balancing Generalized Propensity Score for continuous treatments.

    Pure-Python port of ``CBPS.Continuous`` from the R ``CBPS`` package
    (Fong, Hazlett & Imai, 2018). The treatment is modelled as
    :math:`t \\mid x \\sim N(x^\\top \\beta, \\sigma^2)` and the parameters are
    estimated by GMM combining the likelihood score with the condition that
    the weighted covariance between covariates and treatment vanishes.

    Parameters
    ----------
    method : {"over", "exact"}, default="over"
        ``"over"`` combines score and balance conditions; ``"exact"`` uses
        the balance (and variance) conditions only.
    twostep : bool, default=True
        Two-step GMM (weighting matrix fixed at the MLE) or continuous
        updating.
    max_iter : int, default=1000
    standardize : bool, default=True
        Normalize ``fitted_weights_`` to sum to one.
    density_kind : {"conditional", "stabilized"}, default="conditional"
        ``"conditional"`` returns :math:`p(t \\mid x)`; ``"stabilized"``
        returns :math:`p(t \\mid x) / p(t)` with :math:`p(t)` the Gaussian
        fitted to the marginal treatment distribution (the reciprocal of the
        unstandardized R weights).

    Attributes
    ----------
    coefficients_ : ndarray of shape (p + 1,)
        Conditional-mean coefficients on the original scale, intercept first.
    sigmasq_ : float
        Residual variance on the original treatment scale.
    fitted_values_ : ndarray of shape (n,)
        Conditional density of the standardized treatment (as R reports).
    fitted_weights_ : ndarray of shape (n,)
    J_, mle_J_ : float
    converged_ : bool

    References
    ----------
    Fong, C., Hazlett, C. and Imai, K. (2018). Covariate Balancing Propensity
    Score for a Continuous Treatment. Annals of Applied Statistics.
    """

    _tags = {
        "backend": "pandas",
        "capability:t_type": ["continuous"],
        "capability:multidimensional_treatment": False,
        "density_kind": "conditional",
    }

    def __init__(
        self,
        method: str = "over",
        twostep: bool = True,
        max_iter: int = 1000,
        standardize: bool = True,
        density_kind: str = "conditional",
    ):
        self.method = method
        self.twostep = twostep
        self.max_iter = max_iter
        self.standardize = standardize
        self.density_kind = density_kind
        super().__init__()

        validate_method(self.method)
        if self.density_kind not in ("conditional", "stabilized"):
            raise ValueError(
                f"Invalid density_kind {self.density_kind!r}. "
                "Expected 'conditional' or 'stabilized'."
            )
        self.set_tags(density_kind=self.density_kind)

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame):
        X_array = covariates_to_array(X)
        t_array = t.iloc[:, 0].to_numpy(dtype=float)

        self.design_ = DesignStandardizer().fit(X_array)
        result = fit_cbps_continuous(
            self.design_.u_,
            t_array,
            method=self.method,
            twostep=self.twostep,
            max_iter=self.max_iter,
            standardize=self.standardize,
        )

        self.coefficients_ = self.design_.coef_to_original(result.beta)
        self.sigmasq_ = float(result.extra["sigmasq"])
        self.t_mean_ = float(result.extra["t_mean"])
        self.t_sd_ = float(result.extra["t_sd"])
        self.fitted_values_ = result.fitted_values
        self.fitted_weights_ = result.weights
        self.J_ = result.J
        self.mle_J_ = result.mle_J
        self.converged_ = result.converged
        self.result_ = result
        return self

    def _predict_density(self, X: pd.DataFrame, t: pd.DataFrame) -> np.ndarray:
        X_array = covariates_to_array(X)
        t_array = t.iloc[:, 0].to_numpy(dtype=float)
        mean = np.column_stack([np.ones(len(X_array)), X_array]) @ self.coefficients_
        density = norm.pdf(t_array, loc=mean, scale=np.sqrt(self.sigmasq_))

        if self.density_kind == "stabilized":
            marginal = norm.pdf(t_array, loc=self.t_mean_, scale=self.t_sd_)
            density = density / np.clip(marginal, np.finfo(float).tiny, None)
        return density.reshape(-1, 1)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"method": "over"}, {"method": "exact", "density_kind": "stabilized"}]
