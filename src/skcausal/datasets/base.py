import functools
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import polars as pl
from skbase.base import BaseObject

__all__ = ["BaseDataset", "split_dataset"]


class BaseDataset(BaseObject):
    _tags = {"object_type": ["dataset"]}

    def load(self):
        return self._load()

    def _load(self):
        raise NotImplementedError("Dataset classes should implement _load")


class BaseSyntheticDataset(BaseDataset):
    """
    Abstract base class for datasets used in continuous treatment effect benchmarking.

    Base classes should implement the following methods:

    * get_covariates
    * get_treatments
    * get_outcomes
    * predict_y
    * pdf_treatments
    """

    def __init__(self, n: int, seed=42):

        self.n = n
        self.seed = seed

        super().__init__()

        self._covariates = None
        self._treatments = None
        self._outcomes = None
        self.train_dataset_ = None
        self.test_dataset_ = None
        self._rng = np.random.default_rng(seed)

    def _load(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Generates a dataset with the specified number of samples.

        Args:
            n (int): Number of samples to generate.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the covariates, treatments, and outcomes.
        """
        return self._covariates, self._treatments, self._outcomes

    def _get_covariates(self) -> np.array:
        """
        Generates covariates for the dataset.

        Parameters
        ----------
        n : int
            Number of covariates to generate.

        Returns
        -------
        np.ndarray
            Array of shape (n, m) containing the generated covariates.
        """
        raise NotImplementedError(
            "get_covariates method not implemented for this dataset."
        )

    def _get_treatments(self, covariates: np.ndarray):
        """
        Generates treatment assignments based on the given covariates.

        Args:
            covariates (np.ndarray): Array of shape (n, m) containing the covariates.

        Returns:
            np.ndarray: Array of shape (n,) containing the treatment assignments.
        """
        raise NotImplementedError(
            "get_treatments method not implemented for this dataset."
        )

    def _get_outcomes(
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

        expected_outcomes = self.predict_y(covariates=covariates, treatments=treatments)
        # Add noise
        return self._inject_outcome_noise(
            expected_outcomes, covariates=covariates, treatments=treatments
        )

    def _inject_outcome_noise(self, expected_outcomes, covariates, treatments):
        return expected_outcomes + self._rng.normal(size=expected_outcomes.shape)

    def predict_y(self, covariates: np.ndarray, treatments: np.ndarray) -> np.ndarray:
        """
        Generates mean outcomes based on the given covariates and treatment assignments.

        Args:
            covariates (np.ndarray): Array of shape (n, m) containing the covariates.
            treatments (np.ndarray): Array of shape (n,) containing the treatment assignments.

        Returns:
            np.ndarray: Array of shape (n,) containing the mean outcomes.

        """
        return self._predict_y(covariates, treatments)

    def _predict_y(self, covariates: np.ndarray, treatments: np.ndarray) -> np.ndarray:
        """
        Generates mean outcomes based on the given covariates and treatment assignments.

        Args:
            covariates (np.ndarray): Array of shape (n, m) containing the covariates.
            treatments (np.ndarray): Array of shape (n,) containing the treatment assignments.

        Returns:
            np.ndarray: Array of shape (n,) containing the mean outcomes.

        """
        raise NotImplementedError("predict_y method not implemented for this dataset.")

    def _prepare(
        self,
        n: int = None,
        seed=42,
    ) -> np.ndarray:
        """
        Precompiles a dataset with the specified number of samples.

        Args:
            n (int): Number of samples to generate.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the covariates, treatments, and outcomes.
        """

        if n is not None:
            self.n = n

        covariates = self._get_covariates()
        treatments = self._get_treatments(covariates)
        outcomes = self._get_outcomes(covariates, treatments)

        treatments = self._to_polars(treatments)

        self._covariates = pl.DataFrame(covariates)
        self._treatments = treatments
        self._outcomes = pl.DataFrame(outcomes)

        return self

    def prepare(self, n: int = None, seed=42):
        return self._prepare(n=n, seed=seed)

    @property
    def is_prepared(self):
        return all(
            dataset is not None
            for dataset in (self._covariates, self._treatments, self._outcomes)
        )

    def retrieve(self, test=False):
        if test:
            raise NotImplementedError(
                "Test split retrieval is not implemented for BaseSyntheticDataset."
            )
        return self.load()

    def _to_polars(self, treatments) -> pl.DataFrame:
        if isinstance(treatments, pl.DataFrame):
            return treatments

        schema = getattr(self, "TREATMENT_SCHEMA", None)
        if schema is None:
            return pl.DataFrame(treatments)

        return pl.DataFrame(treatments, schema=schema)

    def predict_adrf(self, covariates: np.ndarray, treatment_list: pl.DataFrame):
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
            outcomes = self.predict_y(covariates, treatments)
            adrf.append(outcomes.mean())

        return np.array(adrf)
