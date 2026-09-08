import numpy as np
import pandas as pd

from skcausal.density.base import BaseDensityEstimator
from skcausal.density.cbps._binary_att import att_probs, fit_cbps_binary_att
from skcausal.density.cbps._common import covariates_to_array, validate_method
from skcausal.density.cbps._design import DesignStandardizer
from skcausal.density.cbps._multinomial import (
    fit_cbps_multinomial,
    multinomial_probs,
)

__all__ = ["CBPSCategorical"]


class CBPSCategorical(BaseDensityEstimator):
    """Covariate Balancing Propensity Score for categorical treatments.

    Pure-Python port of the parametric routines behind ``CBPS()`` in the R
    package of the same name (Imai & Ratkovic, 2014): the binary treatment
    model with ATE or ATT weights and the multinomial-logit model for
    multi-valued treatments. Propensity scores are estimated by GMM using
    both the likelihood score and covariate-balance moment conditions.

    ``predict_density`` returns the fitted probability of the observed
    treatment level, :math:`P(T = t_i \\mid x_i)`. ``predict_proba`` returns
    probabilities for every level in ``classes_`` order.

    Parameters
    ----------
    method : {"over", "exact"}, default="over"
        ``"over"`` fits the over-identified model that combines the score and
        balance conditions; ``"exact"`` uses the balance conditions only.
    att : {0, 1, 2}, default=0
        ``0`` estimates ATE weights. ``1`` estimates ATT weights treating the
        second level as the treatment; ``2`` treats the first level as the
        treatment. Only available for binary treatments.
    twostep : bool, default=True
        Use the two-step GMM estimator (weighting matrix fixed at the
        maximum-likelihood start). ``False`` uses continuous updating.
    max_iter : int, default=1000
        Maximum optimizer iterations.
    standardize : bool, default=True
        Normalize ``fitted_weights_`` to sum to one within each level.

    Attributes
    ----------
    classes_ : ndarray
        Treatment levels; the first is the baseline.
    coefficients_ : ndarray of shape (p + 1, n_classes - 1)
        Logit coefficients on the original covariate scale, intercept first.
    fitted_values_ : ndarray of shape (n, n_classes)
    fitted_weights_ : ndarray of shape (n,)
    J_, mle_J_ : float
        GMM objective at the optimum and at the MLE start.
    converged_ : bool

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B.
    """

    _tags = {
        "backend": "pandas",
        "capability:t_type": ["categorical"],
        "capability:multidimensional_treatment": False,
        "density_kind": "conditional",
    }

    def __init__(
        self,
        method: str = "over",
        att: int = 0,
        twostep: bool = True,
        max_iter: int = 1000,
        standardize: bool = True,
    ):
        self.method = method
        self.att = att
        self.twostep = twostep
        self.max_iter = max_iter
        self.standardize = standardize
        super().__init__()

        validate_method(self.method)
        if self.att not in (0, 1, 2):
            raise ValueError(f"att must be 0, 1 or 2, got {self.att!r}.")

    # ------------------------------------------------------------------ fit
    def _fit(self, X: pd.DataFrame, t: pd.DataFrame):
        X_array = covariates_to_array(X)
        labels = t.iloc[:, 0]
        self.classes_ = self._levels(labels)
        codes = self._encode(labels)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("Treatment must take more than one value.")
        if self.att != 0 and n_classes != 2:
            raise ValueError("att is only supported for binary treatments.")

        self.design_ = DesignStandardizer().fit(X_array)
        U = self.design_.u_

        if self.att != 0:
            treated = (codes == 1) if self.att == 1 else (codes == 0)
            result = fit_cbps_binary_att(
                U,
                treated.astype(float),
                method=self.method,
                twostep=self.twostep,
                max_iter=self.max_iter,
                standardize=self.standardize,
            )
            beta = result.beta if self.att == 1 else -result.beta
            self.beta_u_ = beta.reshape(-1, 1)
            self.fitted_values_ = self._proba_from_u(U)
        else:
            T = np.eye(n_classes)[codes]
            result = fit_cbps_multinomial(
                U,
                T,
                method=self.method,
                twostep=self.twostep,
                max_iter=self.max_iter,
                standardize=self.standardize,
                mle_fallback=n_classes > 2,
            )
            self.beta_u_ = result.beta
            self.fitted_values_ = result.fitted_values

        self.coefficients_ = self.design_.coef_to_original(self.beta_u_)
        self.fitted_weights_ = result.weights
        self.J_ = result.J
        self.mle_J_ = result.mle_J
        self.converged_ = result.converged
        self.result_ = result
        return self

    # -------------------------------------------------------------- predict
    def _proba_from_u(self, U: np.ndarray) -> np.ndarray:
        if self.att != 0:
            p_second = att_probs(U, self.beta_u_[:, 0])
            return np.column_stack([1.0 - p_second, p_second])
        return multinomial_probs(U, self.beta_u_.ravel(order="F"))

    def predict_proba(self, X) -> np.ndarray:
        """Return ``P(T = level | x)`` for every level in ``classes_`` order."""
        X = self._check_and_transform_X(X)
        return self._proba_from_u(self.design_.transform(covariates_to_array(X)))

    def _predict_density(self, X: pd.DataFrame, t: pd.DataFrame) -> np.ndarray:
        probs = self._proba_from_u(self.design_.transform(covariates_to_array(X)))
        codes = self._encode(t.iloc[:, 0])
        return probs[np.arange(len(codes)), codes].reshape(-1, 1)

    # -------------------------------------------------------------- levels
    @staticmethod
    def _levels(labels: pd.Series) -> np.ndarray:
        if isinstance(labels.dtype, pd.CategoricalDtype):
            present = set(labels.unique().tolist())
            return np.asarray([c for c in labels.cat.categories if c in present])
        return np.asarray(sorted(labels.unique().tolist()))

    def _encode(self, labels: pd.Series) -> np.ndarray:
        lookup = {level: index for index, level in enumerate(self.classes_.tolist())}
        try:
            return np.asarray([lookup[value] for value in labels.tolist()], dtype=int)
        except KeyError as error:
            raise ValueError(
                f"Treatment level {error.args[0]!r} was not seen during fit."
            ) from None

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"method": "over"}, {"method": "exact", "twostep": False}]
