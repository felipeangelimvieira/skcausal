"""Shared base for datasets built on a real supervised table.

The model-induced-confounding datasets start from a supervised sample ``(X, y)``
and derive both the treatment and the response surface from a model fitted on
it. Obtaining and preparing that sample is the same work in every such dataset,
so it lives here once.

The source is any :class:`~skcausal.datasets.real.supervised.BaseSupervisedDataset`.
That keeps the loading decision where it belongs, in the source object, and
means these datasets inherit every supervised source that exists now or later
without changing.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.preprocessing import SplineTransformer

from skcausal.datasets.base import BaseKnownResponseDataset
from skcausal.datasets.real.supervised import BaseSupervisedDataset

__all__ = ["BaseSemiSyntheticDataset"]


def _as_matrix(values, *, name: str, expected_width: int | None = None):
    """Return ``values`` as a float ``(n_samples, width)`` array."""

    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} expects a 2D array-like input.")
    if expected_width is not None and array.shape[1] != expected_width:
        raise ValueError(
            f"{name} expects inputs with shape (n_samples, {expected_width})."
        )
    return array


def _as_1d(values, *, name: str, dtype=None):
    """Return ``values`` as a one-dimensional array, unwrapping a single column."""

    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _standardize(values: np.ndarray):
    """Center and scale ``values``; a zero scale is replaced by one."""

    values = np.asarray(values, dtype=float)
    scales = values.std(axis=0, ddof=0)
    return (values - values.mean(axis=0)) / np.where(scales == 0.0, 1.0, scales)


def _fit_random_spline(values, *, rng, n_knots: int, degree: int):
    """Draw a random centered B-spline effect over ``values``.

    The basis is unsupervised: ``SplineTransformer`` only reads the range of
    ``values`` to place its knots. The shape of the curve comes from the
    coefficients, drawn as :math:`N(0, 1/K)` so that the magnitude of the
    random curve does not grow with the number of basis functions.

    Returns ``(transformer, coefficients, offset)``. A constant input has no
    curve to fit, and is reported as a ``None`` transformer.
    """

    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if np.isclose(float(values.min()), float(values.max())):
        return None, np.zeros(1, dtype=float), 0.0

    unique_count = np.unique(np.round(values[:, 0], decimals=12)).size
    transformer = SplineTransformer(
        n_knots=min(n_knots, max(2, unique_count)),
        degree=degree,
        include_bias=False,
        extrapolation="continue",
    )
    basis = transformer.fit_transform(values)
    coefficients = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(max(1, basis.shape[1])),
        size=basis.shape[1],
    )
    return transformer, coefficients, float((basis @ coefficients).mean())


def _evaluate_random_spline(values, *, transformer, coefficients, offset):
    """Evaluate the curve drawn by :func:`_fit_random_spline`, centered."""

    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if transformer is None:
        return np.zeros(values.shape[0], dtype=float)
    return transformer.transform(values) @ coefficients - offset


def _random_level_effects(labels, levels, *, rng):
    """Draw one random effect per level, centered over the realized ``labels``.

    A class label carries no order for a curve to follow, so the categorical
    counterpart of a spline is one independent offset per level.
    """

    raw_effects = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(max(1, len(levels))),
        size=len(levels),
    )
    lookup = dict(zip(levels, raw_effects))
    observed = np.array([lookup[label] for label in labels], dtype=float)
    offset = float(observed.mean()) if observed.size else 0.0
    return {level: float(effect - offset) for level, effect in lookup.items()}


class BaseSemiSyntheticDataset(BaseKnownResponseDataset):
    """Base for datasets whose covariates and target come from a supervised source.

    Subclasses receive standardized covariates from :meth:`_get_covariates` and
    decide what to do with the target through :meth:`_prepare_target`.
    Everything about *where* the supervised sample comes from is the source
    object's business, not theirs.

    The base is :class:`~skcausal.datasets.base.BaseKnownResponseDataset` and
    not :class:`~skcausal.datasets.base.BaseSyntheticDataset`: a semi-synthetic
    dataset has a known response surface, so it is benchmarkable, but its
    covariates are real and its sample size is the source's — there is no ``n``
    for a caller to choose.

    Parameters
    ----------
    dataset : BaseSupervisedDataset
        Source of the supervised sample. Its covariates become the features and
        its single outcome column becomes the target;
        :class:`~skcausal.datasets.real.sklearn.SklearnDataset`,
        :class:`~skcausal.datasets.real.supervised.SupervisedFrame` and
        :class:`~skcausal.datasets.real.openml.OpenMLDataset` are the usual
        choices. A causal dataset is not accepted: it already carries a
        treatment, and these datasets induce their own.
    random_state : int, default=42
        Seed for the derived data-generating process.

    Notes
    -----
    Subclasses declare the supervised task they need in the ``source_task``
    tag; a source whose own ``task`` tag says otherwise is refused at
    construction.
    """

    def __init__(self, dataset, random_state: int = 42):
        if not isinstance(dataset, BaseSupervisedDataset):
            raise TypeError(
                "dataset must be a BaseSupervisedDataset instance, for example "
                "SklearnDataset, SupervisedFrame or OpenMLDataset. Got "
                f"{type(dataset).__name__}."
            )

        # A source that cannot name its task before loading (OpenML's
        # target_kind="auto") tags None, which is not a mismatch: the load-time
        # validation in BaseSupervisedDataset still applies.
        expected = self.get_tag("source_task", None, raise_error=False)
        source_task = dataset.get_tag("task", None, raise_error=False)
        if expected is not None and source_task not in (None, expected):
            raise ValueError(
                f"{type(self).__name__} needs a {expected} source, but "
                f"{type(dataset).__name__} is tagged task={source_task!r}."
            )

        self.dataset = dataset
        super().__init__(random_state=random_state)

    def _prepare_target(self, outcome_frame):
        """Return the source outcome as a 1-D array and store the derived form.

        The returned array is only used to check that the source released one
        target per covariate row. Whatever the subclass actually needs — a
        standardized regression target, a label vector — it stores on itself.
        """

        raise NotImplementedError(
            "Supervised-derived datasets should implement _prepare_target."
        )

    def _load_supervised(self):
        """Return the raw covariate matrix, its column names, and the target."""

        covariates, outcomes = self.dataset.clone().load_covariates_and_outcome()

        outcome_frame = self._coerce_backend_frame(outcomes, backend="polars")
        if outcome_frame.width != 1:
            raise ValueError(
                f"{type(self).__name__} requires a source with exactly one outcome "
                f"column; got {outcome_frame.width}."
            )

        covariate_frame = self._coerce_backend_frame(covariates, backend="polars")
        covariate_array = _as_matrix(
            covariate_frame, name=f"{type(self).__name__} covariates"
        )
        target_array = self._prepare_target(outcome_frame)

        if covariate_array.shape[0] != target_array.shape[0]:
            raise ValueError(
                "The source must provide covariates and targets with the same "
                "number of rows."
            )
        return covariate_array, list(covariate_frame.columns), target_array

    def _get_covariates(self) -> np.ndarray:
        covariate_array, columns, _ = self._load_supervised()

        self.n = covariate_array.shape[0]
        self._n_features = covariate_array.shape[1]
        self.covariate_columns_ = columns or [
            f"x{i}" for i in range(covariate_array.shape[1])
        ]
        return _standardize(covariate_array)

    def _clone_estimator(self, estimator):
        """Clone ``estimator`` and seed an unset top-level ``random_state``."""

        estimator = clone(estimator)
        params = estimator.get_params(deep=False)
        if "random_state" in params and params["random_state"] is None:
            estimator.set_params(random_state=self.random_state)
        return estimator

    def _inject_outcome_noise(self, expected_outcomes, covariates, treatments):
        expected = np.asarray(expected_outcomes, dtype=float).reshape(-1, 1)
        return expected + self._rng.normal(
            loc=0.0,
            scale=self.outcome_noise_scale,
            size=expected.shape[0],
        ).reshape(-1, 1)

    def _covariate_frame(self, covariates: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {
                column: covariates[:, index]
                for index, column in enumerate(self.covariate_columns_)
            }
        )

    def _treatment_frame(self, treatments: np.ndarray) -> pl.DataFrame:
        # The declared column_types decide the dtype: _to_polars runs
        # enforce_dtypes, which casts floats to Float64 and labels to Categorical.
        return self._to_polars(pl.DataFrame({"t": np.asarray(treatments).reshape(-1)}))

    def _outcome_frame(self, outcomes: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({"y": np.asarray(outcomes, dtype=float).reshape(-1)})

    def _prepare(self):
        covariates = self._get_covariates()
        treatments = self._get_treatments(covariates)
        outcomes = self._get_outcomes(covariates, treatments)

        self._covariates = self._covariate_frame(covariates)
        self._treatments = self._treatment_frame(treatments)
        self._outcomes = self._outcome_frame(outcomes)
        self.n = self._covariates.height

        return self
