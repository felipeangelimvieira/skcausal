from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skcausal.datasets.base import BaseDataset, BaseSyntheticDataset

__all__ = ["BaseSemiSyntheticDataset"]


@dataclass(frozen=True)
class _CausalScores:
    confounders: np.ndarray
    prognostic: np.ndarray
    treatment_only: np.ndarray
    modifiers: np.ndarray


@dataclass
class _CausalScoreMap:
    columns: tuple[str, ...]
    encoder: ColumnTransformer
    pair_indices: np.ndarray
    projections: dict[str, np.ndarray]
    means: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]

    def transform(self, X):
        frame = _covariates_to_pandas(X, self.columns)
        encoded = np.asarray(self.encoder.transform(frame), dtype=float)
        library = _nonlinear_library(encoded, self.pair_indices)
        return _CausalScores(
            **{
                name: (library @ self.projections[name] - self.means[name])
                / self.scales[name]
                for name in self.projections
            }
        )


@dataclass(frozen=True)
class _BaselineSurface:
    confounder_coefficients: np.ndarray
    prognostic_coefficients: np.ndarray
    mean: float
    scale: float
    signal_mean: float
    signal_scale: float

    def evaluate(self, scores):
        confounding = scores.confounders @ self.confounder_coefficients
        prognostic = scores.prognostic @ self.prognostic_coefficients
        mu0 = (confounding + prognostic - self.mean) / self.scale
        signal = (confounding - self.signal_mean) / self.signal_scale
        return mu0, signal


def _covariates_to_pandas(X, columns=None):
    if isinstance(X, np.ndarray):
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("NumPy covariates must be a two-dimensional array.")
        if columns is None:
            columns = tuple(f"x{index}" for index in range(array.shape[1]))
        elif array.shape[1] != len(columns):
            raise ValueError("NumPy covariates must match the frozen column width.")
        return pd.DataFrame(array, columns=columns)

    if isinstance(X, pl.DataFrame):
        frame = X.to_pandas()
    elif isinstance(X, pd.DataFrame):
        frame = X.copy()
    else:
        raise TypeError(
            "Covariates must be a NumPy array, pandas DataFrame, or Polars DataFrame."
        )

    if columns is not None and tuple(frame.columns) != tuple(columns):
        raise ValueError("Covariate columns must match the frozen columns exactly.")
    return frame


def _nonlinear_library(encoded, pair_indices):
    encoded = np.asarray(encoded, dtype=float)
    pair_indices = np.asarray(pair_indices, dtype=int)
    if pair_indices.size:
        products = encoded[:, pair_indices[:, 0]] * encoded[:, pair_indices[:, 1]]
    else:
        products = np.empty((encoded.shape[0], 0), dtype=float)
    return np.column_stack(
        (
            encoded,
            encoded**2,
            np.sin(encoded),
            (encoded > 0).astype(float),
            products,
        )
    )


def _fit_causal_score_map(X, rng):
    frame = _covariates_to_pandas(X)
    columns = tuple(frame.columns)
    numeric_columns = [
        name for name in columns if pd.api.types.is_numeric_dtype(frame[name])
    ]
    categorical_columns = [name for name in columns if name not in numeric_columns]

    for name in numeric_columns:
        observed = frame[name].dropna().to_numpy(dtype=float)
        if observed.size == 0 or not np.isfinite(observed).all():
            raise ValueError(
                f"Numeric column {name!r} must contain a finite observed value."
            )

    for name in categorical_columns:
        values = frame[name].astype(object)
        frame[name] = values.where(values.notna(), np.nan)

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="__missing__"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("Covariates must include at least one column.")

    encoder = ColumnTransformer(transformers)
    encoded = np.asarray(encoder.fit_transform(frame), dtype=float)
    n_encoded = encoded.shape[1]
    candidates = np.array(
        [
            (left, right)
            for left in range(n_encoded)
            for right in range(left + 1, n_encoded)
        ],
        dtype=int,
    ).reshape(-1, 2)
    n_pairs = min(16, len(candidates))
    if n_pairs:
        pair_indices = candidates[
            rng.choice(len(candidates), size=n_pairs, replace=False)
        ]
    else:
        pair_indices = np.empty((0, 2), dtype=int)
    library = _nonlinear_library(encoded, pair_indices)

    role_dimensions = {
        "confounders": 3,
        "prognostic": 2,
        "treatment_only": 2,
        "modifiers": 2,
    }
    projections = {}
    means = {}
    scales = {}
    for name, dimension in role_dimensions.items():
        projection = np.zeros((library.shape[1], dimension))
        for index in range(dimension):
            selected = rng.choice(
                library.shape[1], size=min(5, library.shape[1]), replace=False
            )
            projection[selected, index] = rng.normal(size=selected.size)
        raw_scores = library @ projection
        scale = raw_scores.std(axis=0)
        scale[scale == 0] = 1.0
        projections[name] = projection
        means[name] = raw_scores.mean(axis=0)
        scales[name] = scale

    score_map = _CausalScoreMap(
        columns=columns,
        encoder=encoder,
        pair_indices=pair_indices,
        projections=projections,
        means=means,
        scales=scales,
    )
    return score_map, score_map.transform(frame)


def _fit_baseline_surface(scores, rng):
    confounder_scale = 1.0 / np.sqrt(scores.confounders.shape[1])
    prognostic_scale = 1.0 / np.sqrt(scores.prognostic.shape[1])
    confounder_coefficients = rng.normal(
        scale=confounder_scale, size=scores.confounders.shape[1]
    )
    prognostic_coefficients = rng.normal(
        scale=prognostic_scale, size=scores.prognostic.shape[1]
    )
    confounding = scores.confounders @ confounder_coefficients
    total = confounding + scores.prognostic @ prognostic_coefficients
    scale = total.std()
    signal_scale = confounding.std()
    return _BaselineSurface(
        confounder_coefficients=confounder_coefficients,
        prognostic_coefficients=prognostic_coefficients,
        mean=float(total.mean()),
        scale=float(scale if scale != 0 else 1.0),
        signal_mean=float(confounding.mean()),
        signal_scale=float(signal_scale if signal_scale != 0 else 1.0),
    )


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
