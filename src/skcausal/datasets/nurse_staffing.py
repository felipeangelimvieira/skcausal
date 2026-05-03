from typing import Literal

import numpy as np
import polars as pl

from skcausal.datasets.base import BaseSyntheticDataset

__all__ = ["NurseStaffing"]

_COVARIATE_COUNT = 4
_COVARIATE_COLUMNS = ["l1", "l2", "l3", "l4"]
_OUTCOME_TYPES = {"binary", "probability", "logit"}
_COVARIATE_OUTCOME_WEIGHTS = np.array([0.2, 0.2, 0.3, -0.1], dtype=float)
_OUTCOME_QUADRATIC_COEFFICIENT = 0.00132


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped_values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped_values))


def _as_covariate_array(covariates) -> np.ndarray:
    if isinstance(covariates, pl.DataFrame):
        covariates = covariates.to_numpy()

    array = np.asarray(covariates, dtype=float)
    if array.ndim == 1:
        if array.size != _COVARIATE_COUNT:
            raise ValueError("NurseStaffing expects four covariates per sample.")
        array = array.reshape(1, -1)

    if array.ndim != 2 or array.shape[1] != _COVARIATE_COUNT:
        raise ValueError(
            "NurseStaffing expects covariates with shape " "(n_samples, 4)."
        )

    return array


def _as_treatment_array(treatments) -> np.ndarray:
    if isinstance(treatments, pl.DataFrame):
        treatments = treatments.to_numpy()

    array = np.asarray(treatments, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)

    if array.ndim != 2 or array.shape[1] != 1:
        raise ValueError(
            "NurseStaffingSimulationDataset expects treatments with shape "
            "(n_samples, 1)."
        )

    return array


class NurseStaffing(BaseSyntheticDataset):
    r"""
    Continuous-treatment simulation from the nurse staffing application paper.

    The covariates satisfy

    .. math::

        L = (L_1, \ldots, L_4)^\top \sim \mathcal{N}(0, I_4).

    The treatment is generated on the interval :math:`[0, 20]` via

    .. math::

        (A / 20) \mid L \sim \operatorname{Beta}(\lambda(L), 1 - \lambda(L)),

    where

    .. math::

        \operatorname{logit}(\lambda(L)) =
        -0.8 + 0.1L_1 + 0.1L_2 - 0.1L_3 + 0.2L_4.

    The outcome surface follows

    .. math::

        \operatorname{logit}(\mu(L, A)) =
        1 + (0.2, 0.2, 0.3, -0.1)L
        + A(0.1 - 0.1L_1 + 0.1L_3 - 0.00132A^2).

    With ``outcome_type="binary"`` the observed outcome is sampled as
    :math:`Y \mid L, A \sim \operatorname{Bernoulli}(\mu(L, A))`, which matches the
    paper's simulation. ``outcome_type="probability"`` returns :math:`\mu(L, A)`
    directly, while ``outcome_type="logit"`` returns the linear predictor.

    References
    ----------
    Edward H. Kennedy, Zongming Ma, Matthew D. McHugh, and Dylan S. Small.
    "Non-parametric methods for doubly robust estimation of continuous
    treatment effects." Journal of the Royal Statistical Society: Series B
    (Statistical Methodology), 79(4):1229-1245, 2017.
    """

    column_types = {"a": "continuous"}
    TREATMENT_MAX = 20.0

    def __init__(
        self,
        n: int = 1000,
        random_state: int = 42,
        outcome_type: Literal["binary", "probability", "logit"] = "binary",
    ):
        if outcome_type not in _OUTCOME_TYPES:
            valid = ", ".join(sorted(_OUTCOME_TYPES))
            raise ValueError(
                f"outcome_type must be one of {{{valid}}}. Got {outcome_type!r}."
            )

        self.outcome_type = outcome_type

        super().__init__(n=n, random_state=random_state)
        self._prepare(self.n)

    def _get_covariates(self) -> pl.DataFrame:
        covariates = self._rng.normal(size=(self.n, _COVARIATE_COUNT))
        return pl.DataFrame(
            {
                column: covariates[:, idx]
                for idx, column in enumerate(_COVARIATE_COLUMNS)
            }
        )

    def _treatment_mean(self, covariates) -> np.ndarray:
        covariate_array = _as_covariate_array(covariates)
        logits = (
            -0.8
            + 0.1 * covariate_array[:, 0]
            + 0.1 * covariate_array[:, 1]
            - 0.1 * covariate_array[:, 2]
            + 0.2 * covariate_array[:, 3]
        )
        return np.clip(_sigmoid(logits), 1e-6, 1.0 - 1e-6)

    def _get_treatments(self, covariates) -> pl.DataFrame:
        treatment_mean = self._treatment_mean(covariates)
        treatments = self.TREATMENT_MAX * self._rng.beta(
            treatment_mean,
            1.0 - treatment_mean,
            size=treatment_mean.shape,
        )
        return self._to_polars(pl.DataFrame({"a": treatments}))

    def _outcome_logit(self, covariates, treatments) -> np.ndarray:
        covariate_array = _as_covariate_array(covariates)
        treatment_array = _as_treatment_array(treatments)

        if covariate_array.shape[0] != treatment_array.shape[0]:
            raise ValueError(
                "NurseStaffingSimulationDataset requires the same number of "
                "covariate and treatment rows."
            )

        treatment_values = treatment_array[:, 0]
        return (
            1.0
            + covariate_array @ _COVARIATE_OUTCOME_WEIGHTS
            + treatment_values
            * (
                0.1
                - 0.1 * covariate_array[:, 0]
                + 0.1 * covariate_array[:, 2]
                - _OUTCOME_QUADRATIC_COEFFICIENT * treatment_values**2
            )
        )

    def _outcome_probability(self, covariates, treatments) -> np.ndarray:
        return _sigmoid(self._outcome_logit(covariates, treatments))

    def _predict_y(self, covariates, treatments) -> np.ndarray:
        if self.outcome_type == "logit":
            return self._outcome_logit(covariates, treatments)
        return self._outcome_probability(covariates, treatments)

    def _get_outcomes(self, covariates, treatments) -> pl.DataFrame:
        if self.outcome_type == "logit":
            outcomes = self._outcome_logit(covariates, treatments)
        else:
            probabilities = self._outcome_probability(covariates, treatments)
            if self.outcome_type == "probability":
                outcomes = probabilities
            else:
                outcomes = self._rng.binomial(1, probabilities)

        return pl.DataFrame({"y": np.asarray(outcomes).reshape(-1)})

    def get_grid(self, n: int = 100) -> pl.DataFrame:
        return self._coerce_treatment_frame(
            pl.DataFrame({"a": np.linspace(0.0, self.TREATMENT_MAX, n)})
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"n": 2000, "random_state": 7}]
