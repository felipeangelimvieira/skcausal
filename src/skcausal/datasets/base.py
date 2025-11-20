import functools
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import polars as pl

__all__ = ["BaseDataset", "split_dataset"]


class BaseDataset(ABC):
    """
    Abstract base class for datasets used in continuous treatment effect benchmarking.

    Base classes should implement the following methods:

    * generate_covariates
    * get_treatments
    * get_outcomes
    * get_mean_outcomes
    * pdf_treatments
    """

    TREATMENT_SCHEMA = None

    def __init__(self):

        self._covariates = None
        self._treatments = None
        self._outcomes = None
        self.train_dataset_ = None
        self.test_dataset_ = None

    @abstractmethod
    def generate_covariates(self, n: int):
        """
        Generates covariates for the dataset.

        Args:
            n (int): Number of covariates to generate.

        Returns:
            np.ndarray: Array of shape (n, m) containing the generated covariates.
        """
        ...

    @abstractmethod
    def get_treatments(self, covariates: np.ndarray) -> np.ndarray:
        """
        Generates treatment assignments based on the given covariates.

        Args:
            covariates (np.ndarray): Array of shape (n, m) containing the covariates.

        Returns:
            np.ndarray: Array of shape (n,) containing the treatment assignments.
        """
        ...

    @abstractmethod
    def get_outcomes(
        self, covariates: np.ndarray, treatments: np.ndarray
    ) -> np.ndarray:
        """
        Generates outcomes based on the given covariates and treatment assignments.

        Args:
            covariates (np.ndarray): Array of shape (n, m) containing the covariates.
            treatments (np.ndarray): Array of shape (n,) containing the treatment assignments.

        Returns:
            np.ndarray: Array of shape (n,) containing the outcomes.
        """
        ...

    @abstractmethod
    def get_mean_outcomes(
        self, covariates: np.ndarray, treatments: np.ndarray
    ) -> np.ndarray:
        """
        Generates mean outcomes based on the given covariates and treatment assignments.

        Args:
            covariates (np.ndarray): Array of shape (n, m) containing the covariates.
            treatments (np.ndarray): Array of shape (n,) containing the treatment assignments.

        Returns:
            np.ndarray: Array of shape (n,) containing the mean outcomes.

        """
        ...

    def get_dataset(self, n: int) -> np.ndarray:
        """
        Generates a dataset with the specified number of samples.

        Args:
            n (int): Number of samples to generate.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the covariates, treatments, and outcomes.
        """
        covariates = self.generate_covariates(n)
        treatments = self.get_treatments(covariates)
        outcomes = self.get_outcomes(covariates, treatments)

        treatments = self._to_polars(treatments)

        return (
            pl.DataFrame(covariates),
            pl.DataFrame(treatments),
            pl.DataFrame(outcomes),
        )

    def _to_polars(self, treatments: np.ndarray) -> np.ndarray:

        return pl.DataFrame(treatments, schema=self.TREATMENT_SCHEMA)

    def prepare(
        self, n: int = None, test_ratio=0, preparation_seed=None, split_seed=None
    ) -> np.ndarray:
        """
        Precompiles a dataset with the specified number of samples.

        Args:
            n (int): Number of samples to generate.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the covariates, treatments, and outcomes.
        """
        self._rng_preparation = np.random.default_rng(preparation_seed)

        covariates, treatments, outcomes = self.get_dataset(n)
        self._covariates = covariates
        self._treatments = treatments
        self._outcomes = outcomes

        self.train_dataset_, self.test_dataset_ = split_dataset(
            self._covariates,
            self._treatments,
            self._outcomes,
            test_ratio,
            seed=split_seed,
        )
        return self

    @property
    def is_prepared(self):
        return self.train_dataset_ is not None and self.test_dataset_ is not None

    def retrieve(self, test=False):
        if test:
            return self.test_dataset_
        return self.train_dataset_

    def get_adrf(self, covariates: np.ndarray, treatment_list: pl.DataFrame):
        """
        Computes the Average Direct Response Function (ADRF) for the given covariates and treatment list.

        Args:
            covariates (np.ndarray): Array of shape (n, m) containing the covariates.
            treatment_list (List[float]): List of treatment values to compute the ADRF for.

        Returns:
            List[float]: List of ADRF values corresponding to each treatment value.
        """

        if isinstance(treatment_list, pl.DataFrame):
            treatment_list = list(treatment_list.to_numpy())

        adrf = []
        for treatment in treatment_list:
            if isinstance(treatment, np.ndarray):
                treatments = treatment.reshape(1, -1)
                treatments = np.tile(treatments, (covariates.shape[0], 1))
            else:
                treatments = np.full(covariates.shape[0], treatment)
            outcomes = self.get_mean_outcomes(covariates, treatments)
            adrf.append(outcomes.mean())

        return np.array(adrf)

    def pdf_treatments(self, treatments, covariates: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "pdf_treatments method not implemented for this dataset."
        )


def split_dataset(
    X: np.ndarray, t: pl.DataFrame, y: np.ndarray, test_ratio: float = 0.2, seed=None
):
    """
    Splits the dataset into training and testing sets.

    Args:
        X (np.ndarray): The covariates of the dataset with shape (n_samples, n_features).
        t (np.ndarray): The treatment assignments with shape (n_samples,).
        y (np.ndarray): The outcomes with shape (n_samples,).
        test_ratio (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        A tuple containing two tuples:
            - The first tuple contains the training covariates, treatments, and outcomes.
            - The second tuple contains the testing covariates, treatments, and outcomes.

    Example:
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        t = np.array([0, 1, 0, 1, 0])
        y = np.array([1, 3, 5, 7, 9])
        train_data, test_data = split_dataset(X, t, y, test_ratio=0.4)
    """
    n = len(X)
    test_size = int(n * test_ratio)
    train_size = n - test_size

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_set = (X[train_indices], t[train_indices], y[train_indices])
    test_set = (X[test_indices], t[test_indices], y[test_indices])

    return train_set, test_set
