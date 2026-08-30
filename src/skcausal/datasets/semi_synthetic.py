from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from skcausal.datasets.base import BaseDataset, BaseSyntheticDataset

__all__ = ["BaseSemiSyntheticDataset"]


class BaseSemiSyntheticDataset(BaseSyntheticDataset, ABC):
    outcome_columns = ("y",)

    def __init__(self, real_dataset: BaseDataset, random_state: int = 42):
        if not isinstance(real_dataset, BaseDataset):
            raise TypeError("real_dataset must be an instance of BaseDataset.")
        self.real_dataset = real_dataset
        super().__init__(n=0, random_state=random_state)
        self._prepare()

    def _prepare(self, n=None):
        if n is not None:
            raise ValueError(
                "BaseSemiSyntheticDataset derives its sample size from real_dataset "
                "and does not accept a sample size."
            )
        self.real_dataset_ = self.real_dataset.clone()
        X, _, _ = self.real_dataset_.load()
        X = self._coerce_backend_frame(X, backend="polars")
        if X.height == 0 or len(set(X.columns)) != X.width:
            raise ValueError(
                "real_dataset must provide nonempty covariates with unique columns."
            )
        self.n = X.height
        structural_seed, sample_seed = np.random.SeedSequence(self.random_state).spawn(
            2
        )
        self._prepare_dgp(X, np.random.default_rng(structural_seed))
        treatment, outcome = self._draw_sample(X, sample_seed)
        self._covariates = X
        self._treatments = treatment
        self._outcomes = outcome
        return self

    def _draw_sample(self, X, seed):
        treatment_seed, outcome_seed = seed.spawn(2)
        treatment = self._to_polars(
            self._sample_treatment(X, np.random.default_rng(treatment_seed))
        )
        mean = self.predict_y(X, treatment)
        outcome = self._coerce_outcome_frame(
            self._sample_outcome(
                mean, X, treatment, np.random.default_rng(outcome_seed)
            )
        )
        return treatment, outcome

    def _coerce_outcome_frame(self, values):
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.shape != (self.n, len(self.outcome_columns)):
            raise ValueError(
                "Outcome samples must match source rows and outcome columns."
            )
        return pl.DataFrame(
            {name: array[:, index] for index, name in enumerate(self.outcome_columns)}
        )

    def _check_and_transform_X(self, covariates):
        return self._coerce_backend_frame(covariates, backend="polars")

    def _check_and_transform_t(self, treatments):
        return self._coerce_treatment_frame(treatments, backend="polars")

    def sample(self, random_state):
        treatment, outcome = self._draw_sample(
            self._covariates, np.random.SeedSequence(random_state)
        )
        return (
            self._coerce_backend_frame(self._covariates),
            self._coerce_backend_frame(
                treatment, column_types=self._get_treatment_column_types()
            ),
            self._coerce_backend_frame(outcome),
        )

    def log_prob(self, X, treatment):
        X = self._check_and_transform_X(X)
        treatment = self._check_and_transform_t(treatment)
        if self._get_n_samples(X) != self._get_n_samples(treatment):
            raise ValueError(
                "log_prob requires X and treatment to have the same number of rows."
            )
        values = np.asarray(self._log_prob(X, treatment), dtype=float).reshape(-1, 1)
        if values.shape[0] != self._get_n_samples(X) or np.isnan(values).any():
            raise ValueError("_log_prob must return one non-NaN value per row.")
        return values

    def get_grid(self, n=100):
        return self._coerce_treatment_frame(self._get_grid(n))

    @abstractmethod
    def _prepare_dgp(self, X, rng):
        raise NotImplementedError

    @abstractmethod
    def _sample_treatment(self, X, rng):
        raise NotImplementedError

    @abstractmethod
    def _log_prob(self, X, treatment):
        raise NotImplementedError

    @abstractmethod
    def _predict_y(self, X, treatment):
        raise NotImplementedError

    @abstractmethod
    def _sample_outcome(self, mean, X, treatment, rng):
        raise NotImplementedError

    @abstractmethod
    def _get_grid(self, n):
        raise NotImplementedError
