"""
Implement datasets in paper [citatio needed]

There are also some variations, which weren't considered on the paper
"""

import numpy as np
from scipy.stats import beta, norm
import polars as pl
from skcausal.datasets.base import BaseDataset

__all__ = [
    "SyntheticDataset2",
    "SyntheticDataset2Discrete",
]


class SyntheticDataset2(BaseDataset):
    """
    Custom synthetic dataset, with variable number of features.

    """

    def __init__(
        self, outcome_noise=0.5, treatment_noise=0.5, init_seed=5, n_features=6
    ):
        self.outcome_noise = outcome_noise
        self.treatment_noise = treatment_noise
        self.n_features = n_features
        self._rng_preparation = np.random.default_rng(init_seed)

        self.projt_ = self._rng_preparation.normal(0, 1, size=(n_features, 1))

    def generate_covariates(self, n: int) -> np.ndarray:
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

        X = self._rng_preparation.normal(0, 1, size=(n, self.n_features))

        return X

    def _get_inner_embedding(self, X):

        # return np.concatenate([X, np.sin(X), np.cos(X)], axis=1)
        return X

    def get_treatments(self, covariates: np.ndarray) -> np.ndarray:
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
        tprime = self._get_tprime(covariates=covariates)
        tprime1 = self._rng_preparation.beta(tprime, 1 - tprime)
        tprime21 = self._rng_preparation.beta(
            tprime * 99, 100 - tprime * 99, size=tprime.shape
        )
        tprime2 = tprime21
        # tprime22 = self._rng_preparation.beta(100,100-tprime*50, size=tprime.shape)
        # p2 = self._rng_preparation.binomial(1, 0.5, size=tprime.shape)
        # tprime2 = tprime21 * p2 + tprime22 * (1 - p2)
        p = self._rng_preparation.binomial(1, 0.5, size=tprime.shape)
        tprime = tprime1 * p + tprime2 * (1 - p)
        return tprime

    def get_outcomes(
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
        mean = self.get_mean_outcomes(covariates, treatments)
        return self._rng_preparation.normal(mean, np.sqrt(self.outcome_noise))

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
        tprime = 1 / (1 + np.exp(-tprime))
        return tprime

    def get_mean_outcomes(
        self, covariates: np.ndarray, treatments: np.ndarray
    ) -> np.ndarray:
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
        _X = self._get_inner_embedding(covariates)
        t = treatments

        tprime = self._get_tprime(covariates=covariates)
        y = -tprime * 5 + 5 * (t - 0.2) ** 2 + -(t**3)
        return y

    def get_grid(self, n: int = 100):
        return pd.DataFrame(np.linspace(0, 1, n), columns=["t_0"]).astype(
            self.TREATMENT_SCHEMA
        )


class SyntheticDataset2Discrete(SyntheticDataset2):
    """Binary-treatment version of :class:`SyntheticDataset2`."""

    TREATMENT_SCHEMA = [
        ("treatment", np.dtype("bool")),
    ]

    def get_treatments(self, covariates: np.ndarray) -> np.ndarray:
        prob = self._get_tprime(covariates=covariates)
        return self._rng_preparation.binomial(1, prob, size=prob.shape).astype(bool)

    def get_grid(self, n: int = 100):
        data = [True, False]
        return pl.DataFrame(data, schema=self.TREATMENT_SCHEMA)
