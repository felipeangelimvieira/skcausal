import numpy as np
import polars as pl

from skcausal.datasets.base import BaseSyntheticDataset

__all__ = ["SyntheticVCNet"]

_COVARIATE_COLUMNS = [f"x{i}" for i in range(1, 7)]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped_values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped_values))


def _as_2d_float_array(values, *, expected_width: int, name: str) -> np.ndarray:
    if isinstance(values, pl.DataFrame):
        values = values.to_numpy()
    elif hasattr(values, "to_numpy"):
        values = values.to_numpy()

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if expected_width == 1:
            array = array.reshape(-1, 1)
        elif array.size == expected_width:
            array = array.reshape(1, -1)
        else:
            raise ValueError(
                f"{name} expects rows with exactly {expected_width} values."
            )

    if array.ndim != 2 or array.shape[1] != expected_width:
        raise ValueError(
            f"{name} expects inputs with shape (n_samples, {expected_width})."
        )

    return array


def _as_treatment_array(values, *, name: str) -> np.ndarray:
    return _as_2d_float_array(values, expected_width=1, name=name)[:, 0]


class SyntheticVCNet(BaseSyntheticDataset):
    r"""Fully synthetic continuous-treatment benchmark from [1].

    The observed covariates satisfy

    .. math::

        X_j \stackrel{\mathrm{iid}}{\sim} \operatorname{Unif}(0, 1),
        \qquad j = 1, \ldots, 6.

    A latent treatment score is sampled as

    .. math::

        \tilde{T} =
        \frac{10\sin(\max(X_1, X_2, X_3)) + \max(X_3, X_4, X_5)^3}
        {1 + (X_1 + X_5)^2}
        + \sin(0.5X_3)(1 + \exp(X_4 - 0.5X_3))
        + X_3^2 + 2\sin(X_4) + 2X_5 - 6.5 + \varepsilon_T,

    with :math:`\varepsilon_T \sim \mathcal{N}(0, 0.25)`, and the observed
    treatment is :math:`T = \operatorname{logit}^{-1}(\tilde{T})`.

    The noiseless response surface used by :meth:`predict_y` is

    .. math::

        Q(T, X) = \cos(2\pi(T - 0.5))
        \left[T^2 + \frac{4\max(X_1, X_6)^3}{1 + 2X_3^2}\sin(X_4)\right].

    Observed outcomes add independent Gaussian noise with variance 0.25. This
    implementation exposes a single sample of size ``n`` and leaves any
    train/test splitting to external split objects.


    References
    ----------
    Nie, L., Ye, M., Liu, Q., & Nicolae, D. (2021).
    Vcnet and functional targeted regularization for learning causal effects of
    `continuous treatments. arXiv preprint arXiv:2103.07861.

    """

    TREATMENT_SCHEMA = pl.Schema({"t": pl.Float64})

    def __init__(
        self,
        n: int = 500,
        random_state: int = 42,
        treatment_noise_variance: float = 0.25,
        outcome_noise_variance: float = 0.25,
    ):
        if n <= 0:
            raise ValueError("n must be positive.")
        if treatment_noise_variance < 0.0 or outcome_noise_variance < 0.0:
            raise ValueError("Noise variances must be non-negative.")

        self.n = n
        self.treatment_noise_variance = treatment_noise_variance
        self.outcome_noise_variance = outcome_noise_variance

        super().__init__(n=self.n, random_state=random_state)
        self._prepare(self.n)

    def _covariate_frame(self, covariates: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {
                column: covariates[:, idx]
                for idx, column in enumerate(_COVARIATE_COLUMNS)
            }
        )

    def _treatment_frame(self, treatments: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {"t": np.asarray(treatments, dtype=float).reshape(-1)},
            schema=self.TREATMENT_SCHEMA,
        )

    def _outcome_frame(self, outcomes: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({"y": np.asarray(outcomes, dtype=float).reshape(-1)})

    def _sample_covariates(self, n_samples: int) -> np.ndarray:
        return self._rng.uniform(0.0, 1.0, size=(n_samples, len(_COVARIATE_COLUMNS)))

    def _treatment_latent_score(self, covariates) -> np.ndarray:
        covariate_array = _as_2d_float_array(
            covariates,
            expected_width=len(_COVARIATE_COLUMNS),
            name=type(self).__name__,
        )
        x1, x2, x3, x4, x5, _ = covariate_array.T
        return (
            (
                10.0 * np.sin(np.maximum.reduce([x1, x2, x3]))
                + np.maximum.reduce([x3, x4, x5]) ** 3
            )
            / (1.0 + (x1 + x5) ** 2)
            + np.sin(0.5 * x3) * (1.0 + np.exp(x4 - 0.5 * x3))
            + x3**2
            + 2.0 * np.sin(x4)
            + 2.0 * x5
            - 6.5
        )

    def _sample_treatments(self, covariates) -> np.ndarray:
        latent_score = self._treatment_latent_score(covariates)
        latent_score = latent_score + self._rng.normal(
            scale=np.sqrt(self.treatment_noise_variance),
            size=latent_score.shape[0],
        )
        return _sigmoid(latent_score)

    def _predict_y(self, covariates, treatments) -> np.ndarray:
        covariate_array = _as_2d_float_array(
            covariates,
            expected_width=len(_COVARIATE_COLUMNS),
            name=type(self).__name__,
        )
        treatment_array = _as_treatment_array(
            treatments,
            name=f"{type(self).__name__} treatments",
        )

        x1 = covariate_array[:, 0]
        x3 = covariate_array[:, 2]
        x4 = covariate_array[:, 3]
        x6 = covariate_array[:, 5]

        return np.cos(2.0 * np.pi * (treatment_array - 0.5)) * (
            treatment_array**2
            + (4.0 * np.maximum(x1, x6) ** 3 / (1.0 + 2.0 * x3**2)) * np.sin(x4)
        )

    def _sample_outcomes(self, covariates, treatments) -> np.ndarray:
        expected_outcomes = self._predict_y(covariates, treatments)
        return expected_outcomes + self._rng.normal(
            scale=np.sqrt(self.outcome_noise_variance),
            size=expected_outcomes.shape[0],
        )

    def _get_covariates(self) -> np.ndarray:
        return self._sample_covariates(self.n)

    def _get_treatments(self, covariates) -> pl.DataFrame:
        return self._treatment_frame(self._sample_treatments(covariates))

    def _get_outcomes(self, covariates, treatments) -> pl.DataFrame:
        return self._outcome_frame(self._sample_outcomes(covariates, treatments))

    def _prepare(self, n: int = None):
        if n is not None:
            self.n = n

        covariates = self._covariate_frame(self._get_covariates())
        treatments = self._get_treatments(covariates)
        outcomes = self._get_outcomes(covariates, treatments)

        self._covariates = covariates
        self._treatments = treatments
        self._outcomes = outcomes

        return self

    def get_grid(self, n: int = 100) -> pl.DataFrame:
        return pl.DataFrame(
            {"t": np.linspace(0.0, 1.0, n)},
            schema=self.TREATMENT_SCHEMA,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"n": 128, "random_state": 7}]
