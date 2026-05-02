"""
Implement datasets in paper [citatio needed]

There are also some variations, which weren't considered on the paper
"""

import numpy as np
import polars as pl

from skcausal.datasets.base import BaseSyntheticDataset

__all__ = [
    "SyntheticDataset2",
    "SyntheticDataset2Discrete",
]


def _as_2d_array(values, *, dtype=float) -> np.ndarray:
    if isinstance(values, pl.DataFrame):
        values = values.to_numpy()

    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped_values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped_values))


class SyntheticDataset2(BaseSyntheticDataset):
    r"""Continuous-treatment synthetic benchmark with latent treatment score.

    The covariates satisfy

    .. math::

        X_j \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, 1),
        \qquad j = 1, \ldots, p,

    where :math:`p =` ``n_features``.

    A Gaussian projection vector

    .. math::

        w \sim \mathcal{N}(0, I_p)

    is sampled once per dataset instance and defines the latent treatment mean

    .. math::

        t'(X) = \operatorname{logit}^{-1}\!\left(\frac{X^\top w}{\sqrt{p}}\right).

    The observed treatment is drawn from a two-component mixture that shares the
    same conditional mean signal:

    .. math::

        T \mid X \sim
        B \cdot \operatorname{Beta}(t'(X), 1 - t'(X))
        + (1 - B) \cdot \operatorname{Beta}(99t'(X), 100 - 99t'(X)),

    where :math:`B \sim \operatorname{Bernoulli}(0.5)` independently for each
    row. The implementation clips :math:`t'(X)` away from 0 and 1 before passing
    it to the beta distributions.

    The noiseless response surface is

    .. math::

        m(X, T) = -5t'(X) + 5(T - 0.2)^2 - T^3.

    Observed outcomes satisfy

    .. math::

        Y \mid X, T \sim \mathcal{N}(m(X, T), \sigma_Y^2),

    with :math:`\sigma_Y^2 =` ``outcome_noise``.
    """

    TREATMENT_SCHEMA = pl.Schema({"t_0": pl.Float64})

    def __init__(
        self,
        n=1000,
        outcome_noise=0.5,
        treatment_noise=0.5,
        random_state=5,
        n_features=6,
    ):
        self.outcome_noise = outcome_noise
        self.treatment_noise = treatment_noise
        self.random_state = random_state
        self.n_features = n_features

        super().__init__(n=n, seed=random_state)
        self.projt_ = self._rng.normal(0, 1, size=(n_features, 1))
        self._prepare(self.n)

    def _get_covariates(self, n: int = None) -> np.ndarray:
        """Generate the X array of covariates.

        Parameters
        ----------
        n : int
            the number of samples

        Returns
        -------
        np.ndarray
            array with covariates, sampled uniformly from [0, 1]
        """

        n = self.n if n is None else n
        X = self._rng.normal(0, 1, size=(n, self.n_features))

        return X

    def _get_inner_embedding(self, X):
        # return np.concatenate([X, np.sin(X), np.cos(X)], axis=1)
        return _as_2d_array(X)

    def _get_treatments(self, covariates: np.ndarray) -> np.ndarray:
        """
        Return the array of treatments.

        Parameters
        ----------
        covariates: np.ndarray
            the array of covariates

        Returns
        -------
        np.ndarray
            the array of treatments
        """
        tprime = np.clip(self._get_tprime(covariates=covariates), 1e-6, 1 - 1e-6)
        tprime1 = self._rng.beta(tprime, 1 - tprime)
        tprime21 = self._rng.beta(tprime * 99, 100 - tprime * 99, size=tprime.shape)
        tprime2 = tprime21
        # tprime22 = self._rng.beta(100,100-tprime*50, size=tprime.shape)
        # p2 = self._rng.binomial(1, 0.5, size=tprime.shape)
        # tprime2 = tprime21 * p2 + tprime22 * (1 - p2)
        p = self._rng.binomial(1, 0.5, size=tprime.shape)
        tprime = tprime1 * p + tprime2 * (1 - p)
        return tprime

    def _get_outcomes(
        self, covariates: np.ndarray, treatments: np.ndarray
    ) -> np.ndarray:
        """
        Get the array of outcomes for the given covariates and treatments.

        Sample $Y \sim \mathcal{N}(u2, \sigma^2)$

        Parameters
        ----------
        covariates: np.ndarray
            array of covariates

        treatments: np.ndarray
            array of treatments

        Returns
        -------
        np.ndarray
            array of outcomes
        """
        mean = self.predict_y(covariates, treatments)
        return self._rng.normal(mean, np.sqrt(self.outcome_noise))

    def _get_tprime(self, covariates: np.ndarray) -> np.ndarray:
        """
        Get the tprime for the given covariates.

        Parameters
        ----------
        covariates: np.ndarray
            array of covariates

        Returns
        -------
        np.ndarray
            array of tprime
        """
        _X = self._get_inner_embedding(covariates)
        tprime = _X @ self.projt_ / np.sqrt(self.n_features)
        tprime = _sigmoid(tprime)
        return tprime

    def _predict_y(self, covariates: np.ndarray, treatments: np.ndarray) -> np.ndarray:
        """
        Get the E[Y|X, T] for the given covariates and treatments.

        Parameters
        ----------
        covariates: np.ndarray
            array of covariates

        treatments: np.ndarray
            array of treatments

        Returns
        -------
        np.ndarray
            array of mean outcomes
        """
        t = _as_2d_array(treatments)
        tprime = self._get_tprime(covariates=covariates)
        y = -tprime * 5 + 5 * (t - 0.2) ** 2 + -(t**3)
        return y

    def get_grid(self, n: int = 100):
        return pl.DataFrame(
            {"t_0": np.linspace(0, 1, n)},
            schema=self.TREATMENT_SCHEMA,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"n": 1000}]


class SyntheticDataset2Discrete(SyntheticDataset2):
    r"""Binary-treatment version of :class:`SyntheticDataset2`.

    The covariates and latent score follow the same mechanism as
    :class:`SyntheticDataset2`:

    .. math::

        X_j \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, 1),
        \qquad
        t'(X) = \operatorname{logit}^{-1}\!\left(\frac{X^\top w}{\sqrt{p}}\right).

    The treatment becomes binary via

    .. math::

        A \mid X \sim \operatorname{Bernoulli}(t'(X)).

    The structural response is still inherited from the parent dataset and is

    .. math::

        m(X, A) = -5t'(X) + 5(A - 0.2)^2 - A^3,

    with observed outcomes sampled as

    .. math::

        Y \mid X, A \sim \mathcal{N}(m(X, A), \sigma_Y^2).
    """

    TREATMENT_SCHEMA = pl.Schema({"treatment": pl.Boolean})

    def _get_treatments(self, covariates: np.ndarray) -> np.ndarray:
        prob = np.clip(self._get_tprime(covariates=covariates), 1e-6, 1 - 1e-6)
        return self._rng.binomial(1, prob, size=prob.shape).astype(bool)

    def get_grid(self, n: int = 100):
        return pl.DataFrame(
            {"treatment": [True, False]},
            schema=self.TREATMENT_SCHEMA,
        )
