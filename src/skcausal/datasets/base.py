from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from skbase.base import BaseObject

from skcausal import config
from skcausal.datatypes import convert, enforce_dtypes

__all__ = [
    "BaseDataset",
    "BaseKnownResponseDataset",
    "BaseSyntheticDataset",
    "BaseTabularDataset",
]

ColumnTypeName = Literal["continuous", "categorical"]
TaskName = Literal["regression", "classification"]


class BaseTabularDataset(BaseObject):
    """A table of covariates and one outcome, in whatever backend is configured.

    This is what every dataset in the package has in common, and it is all it
    is: the frame coercion, the backend choice, and the guarantee that
    covariates and an outcome can be obtained. Whether the table also carries a
    treatment is the subclass's business — :class:`BaseDataset` says yes and
    releases the causal triple, :class:`~skcausal.datasets.real.supervised.BaseSupervisedDataset`
    says no and releases ``(covariates, outcome)``.

    The split exists so that a supervised benchmark does not have to pretend to
    be a causal problem. Consumers that only need covariates and an outcome —
    the model-induced datasets, which generate their own treatment — accept
    this type and take either kind of source.
    """

    # "task" is the supervised task the table poses: "regression" for a numeric
    # outcome, "classification" for labels, None when the source cannot tell
    # before loading. Consumers that only accept one kind check it.
    _tags = {"backend": "polars", "task": None}

    def load_covariates_and_outcome(self):
        """Return ``(covariates, outcome)``, whatever else this dataset holds."""

        raise NotImplementedError(
            "Tabular dataset classes should implement load_covariates_and_outcome."
        )

    def _coerce_backend_frame(self, value, *, column_types=None, backend=None):
        column_names = list(column_types) if column_types else None
        if isinstance(value, np.ndarray):
            frame = self._frame_from_array(value, column_names=column_names)
        else:
            if isinstance(value, pd.Series):
                if column_names is not None:
                    if len(column_names) != 1:
                        raise ValueError(
                            "Series inputs can only be coerced when exactly one "
                            "column name is configured."
                        )
                    value = value.rename(column_names[0])
                value = value.to_frame()

            if isinstance(value, pl.Series):
                if column_names is not None:
                    if len(column_names) != 1:
                        raise ValueError(
                            "Series inputs can only be coerced when exactly one "
                            "column name is configured."
                        )
                    value = value.alias(column_names[0])
                frame = value.to_frame()
                return self._finalize_output_frame(
                    frame,
                    column_types=column_types,
                    backend=backend,
                )

            if isinstance(value, pd.DataFrame) or isinstance(value, pl.DataFrame):
                frame = value
            else:
                raise TypeError(
                    "Datasets must expose covariates, treatments, and outcomes as "
                    "numpy arrays or dataframe-like objects. "
                    f"Got {type(value).__name__}."
                )

        frame = self._ensure_column_names(frame, column_names=column_names)
        return self._finalize_output_frame(
            frame,
            column_types=column_types,
            backend=backend,
        )

    def _finalize_output_frame(self, frame, *, column_types=None, backend=None):
        target_backend = self._get_output_backend() if backend is None else backend
        frame = convert(frame, target_backend)
        if column_types:
            frame = enforce_dtypes(frame, column_types=column_types)
        return frame

    def _get_output_backend(self) -> str:
        return config.get_default_backend(fallback=self.get_tag("backend"))

    def _frame_from_array(self, value, *, column_names=None):
        array = np.asarray(value)
        if array.ndim == 0:
            raise ValueError("Expected array-like input with at least one dimension.")
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError(
                "Expected array-like input that can be coerced to a 2D frame, "
                f"but got shape {array.shape}."
            )

        if column_names is None:
            return pl.DataFrame(array)

        if array.shape[1] != len(column_names):
            raise ValueError(
                "Configured column names do not match the array width: "
                f"expected {len(column_names)} columns but got {array.shape[1]}."
            )

        return pl.DataFrame(
            {column: array[:, idx] for idx, column in enumerate(column_names)}
        )

    def _ensure_column_names(self, frame, *, column_names=None):
        if column_names is None:
            return frame

        current_columns = list(frame.columns)
        if current_columns == column_names:
            return frame

        if len(current_columns) != len(column_names):
            raise ValueError(
                "Configured column names do not match the dataframe width: "
                f"expected {len(column_names)} columns but got {len(current_columns)}."
            )

        rename_map = dict(zip(current_columns, column_names))
        if isinstance(frame, pd.DataFrame):
            return frame.rename(columns=rename_map)
        return frame.rename(rename_map)


class BaseDataset(BaseTabularDataset):
    """A causal problem: covariates, the treatment they received, and outcomes.

    ``column_types`` declares the measurement kind of each treatment column, so
    that the treatment frame is released with the dtypes an estimator expects.
    """

    _tags = {"object_type": ["dataset"]}
    column_types: dict[str, ColumnTypeName] = {}

    def load(self):
        covariates, treatments, outcomes = self._load()
        return (
            self._coerce_backend_frame(covariates),
            self._coerce_backend_frame(
                treatments,
                column_types=self._get_treatment_column_types(),
            ),
            self._coerce_backend_frame(outcomes),
        )

    def load_covariates_and_outcome(self):
        covariates, _, outcomes = self.load()
        return covariates, outcomes

    def _load(self):
        raise NotImplementedError("Dataset classes should implement _load")

    def _coerce_treatment_frame(self, value, *, backend=None):
        return self._coerce_backend_frame(
            value,
            column_types=self._get_treatment_column_types(),
            backend=backend,
        )

    def _get_treatment_column_types(self) -> dict[str, ColumnTypeName]:
        return dict(self.column_types)


class BaseKnownResponseDataset(BaseDataset):
    r"""A causal dataset whose true response surface is known.

    This is what makes a dataset benchmarkable: ``predict_y`` exposes the
    noiseless :math:`E[Y \mid X, T]` the outcomes were drawn from, and
    ``predict_curve`` averages it over a treatment grid, so an estimated curve
    can be scored against truth.

    How many rows there are is deliberately not part of that contract. A
    dataset that draws its own sample takes ``n`` from the caller — that is
    :class:`BaseSyntheticDataset`. One whose covariates come from somewhere
    else takes the row count from that source, and has no ``n`` to be given.
    Both are benchmarkable, so both live under this class rather than under the
    one that owns a sample size.

    Subclasses implement ``_get_covariates``, ``_get_treatments`` and
    ``_predict_y``, and call ``_prepare`` once the attributes those hooks read
    have been assigned.
    """

    def __init__(self, random_state: int = 42):

        self.random_state = random_state

        super().__init__()

        self._covariates = None
        self._treatments = None
        self._outcomes = None
        self._rng = np.random.default_rng(random_state)

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

    def _check_and_transform_X(self, covariates):
        if isinstance(covariates, np.ndarray):
            return covariates
        return self._coerce_backend_frame(
            covariates,
            backend=self.get_tag("backend"),
        )

    def _check_and_transform_t(self, treatments):
        if isinstance(treatments, np.ndarray):
            return treatments
        return self._coerce_treatment_frame(
            treatments,
            backend=self.get_tag("backend"),
        )

    def _get_n_samples(self, value) -> int:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                raise ValueError(
                    "Expected array-like input with at least one dimension."
                )
            return value.shape[0]
        if isinstance(value, pd.DataFrame):
            return len(value)
        if isinstance(value, pl.DataFrame):
            return value.height
        raise TypeError(
            f"Cannot infer number of samples from object of type {type(value).__name__}."
        )

    def _coerce_individual_predictions(self, predictions, *, n_samples):
        predictions = np.asarray(predictions, dtype=float)
        if predictions.ndim == 0:
            raise ValueError("predict_y must return at least one prediction.")
        if predictions.shape[0] != n_samples:
            raise ValueError("predict_y must return one prediction per treatment row.")
        if predictions.ndim == 1:
            return predictions.reshape(-1, 1)
        return predictions.reshape(n_samples, -1)

    def _coerce_curve_predictions(self, predictions, *, n_rows):
        predictions = np.asarray(predictions, dtype=float)
        if predictions.ndim == 0:
            raise ValueError("predict_curve must return at least one prediction.")
        if predictions.shape[0] != n_rows:
            raise ValueError(
                "predict_curve must return one prediction per requested treatment row."
            )
        if predictions.ndim == 1:
            return predictions

        predictions = predictions.reshape(n_rows, -1)
        if predictions.shape[1] == 1:
            return predictions[:, 0]
        return predictions

    def predict_y(self, covariates: np.ndarray, treatments: np.ndarray) -> np.ndarray:
        """
        Generates mean outcomes based on the given covariates and treatment assignments.

        Args:
            covariates (np.ndarray): Array of shape (n, m) containing the covariates.
            treatments (np.ndarray): Array of shape (n,) containing the treatment assignments.

        Returns:
            np.ndarray: Array of shape (n,) containing the mean outcomes.

        """
        covariates = self._check_and_transform_X(covariates)
        treatments = self._check_and_transform_t(treatments)

        n_samples = self._get_n_samples(treatments)
        if self._get_n_samples(covariates) != n_samples:
            raise ValueError(
                "predict_y requires covariates and treatments to have the same "
                "number of rows."
            )

        predictions = self._predict_y(covariates, treatments)
        return self._coerce_individual_predictions(predictions, n_samples=n_samples)

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

    def _prepare(self):
        """Run the data-generating process once and store the realized sample.

        Returns
        -------
        self
        """

        covariates = self._get_covariates()
        treatments = self._get_treatments(covariates)
        outcomes = self._get_outcomes(covariates, treatments)

        treatments = self._to_polars(treatments)

        self._covariates = pl.DataFrame(covariates)
        self._treatments = treatments
        self._outcomes = pl.DataFrame(outcomes)

        return self

    def prepare(self):
        return self._prepare()

    @property
    def is_prepared(self):
        return all(
            dataset is not None
            for dataset in (self._covariates, self._treatments, self._outcomes)
        )

    def _to_polars(self, treatments) -> pl.DataFrame:
        if isinstance(treatments, pl.DataFrame):
            return enforce_dtypes(
                self._ensure_column_names(
                    treatments,
                    column_names=list(self._get_treatment_column_types()) or None,
                ),
                column_types=self._get_treatment_column_types() or None,
            )

        frame = self._coerce_treatment_frame(treatments, backend="polars")
        polars_frame = convert(frame, "polars")
        if self._get_treatment_column_types():
            return enforce_dtypes(
                polars_frame,
                column_types=self._get_treatment_column_types(),
            )
        return polars_frame

    def predict_curve(self, covariates: np.ndarray, treatment_grid: pl.DataFrame):
        """Compute the average response at each row of a treatment grid.

        For each requested treatment row, this evaluates ``predict_y`` across the
        full covariate sample and returns the mean response. This is the smooth
        response curve associated with the dataset's noiseless outcome model.
        """

        covariates = self._check_and_transform_X(covariates)
        treatment_grid = self._check_and_transform_t(treatment_grid)

        covariate_count = (
            covariates.height
            if isinstance(covariates, pl.DataFrame)
            else covariates.shape[0]
        )

        if isinstance(treatment_grid, pl.DataFrame):
            treatment_rows = (
                treatment_grid.slice(row_index, 1)
                for row_index in range(treatment_grid.height)
            )
        else:
            treatment_array = np.asarray(treatment_grid)
            if treatment_array.ndim == 0:
                treatment_rows = [treatment_array.item()]
            else:
                treatment_rows = list(treatment_array)

        curve = []
        for treatment in treatment_rows:
            if isinstance(treatment, pl.DataFrame):
                # Keep mixed continuous/categorical column dtypes intact.
                tiled = treatment.select(
                    pl.all().repeat_by(covariate_count).explode()
                )
            else:
                row = np.asarray(treatment)
                if row.ndim == 0:
                    tiled = np.full(covariate_count, row.item())
                else:
                    tiled = np.tile(row.reshape(1, -1), (covariate_count, 1))

            curve.append(
                np.asarray(self.predict_y(covariates, tiled), dtype=float).mean(axis=0)
            )

        return self._coerce_curve_predictions(curve, n_rows=len(curve))

    def predict(self, covariates: np.ndarray, treatment_list: pl.DataFrame):
        """
        Computes the Average Direct Response Function (ADRF) for the given covariates and treatment list.

        Args:
            covariates (np.ndarray): Array of shape (n, m) containing the covariates.
            treatment_list (List[float]): List of treatment values to compute the ADRF for.

        Returns:
            List[float]: List of ADRF values corresponding to each treatment value.
        """

        return self.predict_curve(covariates=covariates, treatment_grid=treatment_list)


class BaseSyntheticDataset(BaseKnownResponseDataset):
    """A data-generating process the caller draws ``n`` rows from.

    Everything the benchmark needs is inherited; what this class adds is the
    sample size, as a constructor argument and as an argument to ``prepare``.
    Use it when the dataset invents its own covariates. When they come from a
    real table instead, subclass :class:`BaseKnownResponseDataset` directly:
    a sample size the caller cannot choose does not belong in the signature.
    """

    def __init__(self, n: int, random_state: int = 42):
        self.n = n
        super().__init__(random_state=random_state)

    def _prepare(self, n: int = None):
        """Generate the sample, optionally resizing it to ``n`` rows first."""

        if n is not None:
            self.n = n
        return super()._prepare()

    def prepare(self, n: int = None):
        return self._prepare(n=n)
