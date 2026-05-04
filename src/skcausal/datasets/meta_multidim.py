"""Meta-dataset that augments a continuous treatment with a binned category.

This wrapper turns any synthetic dataset with exactly one continuous treatment
column into a mixed-treatment dataset with two treatment columns:

* the original continuous treatment, and
* a categorical column obtained by percentile-binning that continuous value.

The categorical column can be partially decorrelated from the original
continuous treatment through ``mutual_info``. When ``mutual_info=1``, the
categorical column is the deterministic percentile bin. When ``mutual_info=0``,
the categorical labels are fully permuted, preserving the marginal bin counts
while breaking the row-wise association with the continuous treatment.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from skcausal.datatypes import collect_column_types, convert
from skcausal.datasets.base import BaseSyntheticDataset
from skcausal.datasets.synthetic2 import SyntheticDataset2

__all__ = ["MetaMultidimDataset"]


def _as_polars_frame(value) -> pl.DataFrame:
    if isinstance(value, pl.DataFrame):
        return value
    return convert(value, "polars")


def _validate_n_categorical_treatments(n_categorical_treatments: int) -> int:
    if (
        not isinstance(n_categorical_treatments, int)
        or isinstance(n_categorical_treatments, bool)
        or n_categorical_treatments < 2
    ):
        raise ValueError(
            "n_categorical_treatments must be an integer greater than or equal to 2."
        )
    return n_categorical_treatments


def _validate_mutual_info(mutual_info: float) -> float:
    value = float(mutual_info)
    if not 0.0 <= value <= 1.0:
        raise ValueError("mutual_info must be between 0 and 1 inclusive.")
    return value


def _validate_nonnegative_float(value, *, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


class MetaMultidimDataset(BaseSyntheticDataset):
    r"""Mixed continuous/categorical treatment wrapper for synthetic datasets.

    The wrapped base dataset must expose exactly one continuous treatment
    column. Let that original treatment be :math:`T`. This meta-dataset adds a
    derived categorical column :math:`C` formed by splitting :math:`T` into
    ``n_categorical_treatments`` percentile bins.

        The structural response augments that of the base dataset:

    .. math::

            m_{\mathrm{meta}}(X, T, C)
            = \gamma(C) \, m_{\mathrm{base}}(X, T).

        Here :math:`\gamma(C)` is a category-specific multiplier sampled once per
        dataset instance, centered around 1.0.

    Parameters
    ----------
        base_dataset : BaseSyntheticDataset
            One-dimensional continuous-treatment synthetic dataset to wrap.
    n_categorical_treatments : int, default=4
            Number of percentile bins used to form the categorical treatment.
    mutual_info : float, default=1.0
            Fraction of rows that keep the deterministic percentile-bin assignment.
            The remaining rows have their labels permuted among themselves.
        categorical_effect_scale : float, default=0.15
            Standard deviation of the Normal distribution used to sample the
            category-specific multiplicative effect around 1.0.
    categorical_column : str, optional
            Name of the derived categorical treatment column. Defaults to
            ``"<continuous_column>_bin"``.
    random_state : int, default=0
            Seed used for both the wrapped dataset and the label permutation step.
    """

    column_types = {"t_0": "continuous", "t_0_bin": "categorical"}

    def __init__(
        self,
        base_dataset: BaseSyntheticDataset,
        n_categorical_treatments: int = 4,
        mutual_info: float = 1.0,
        categorical_effect_scale: float = 0.15,
        categorical_column: str | None = None,
        random_state: int = 0,
    ):
        if not isinstance(base_dataset, BaseSyntheticDataset):
            raise TypeError("base_dataset must be an instance of BaseSyntheticDataset.")

        self.base_dataset = base_dataset
        self.n_categorical_treatments = _validate_n_categorical_treatments(
            n_categorical_treatments
        )
        self.mutual_info = _validate_mutual_info(mutual_info)
        self.categorical_effect_scale = _validate_nonnegative_float(
            categorical_effect_scale,
            name="categorical_effect_scale",
        )
        self.categorical_column = categorical_column
        self._resolved_base_dataset = base_dataset

        super().__init__(n=int(base_dataset.n), random_state=random_state)
        self._prepare()

    def _make_base_dataset(self) -> BaseSyntheticDataset:
        return self._resolved_base_dataset.clone()

    def _prepare(self, n: int = None):
        if n is not None:
            raise ValueError(
                "MetaMultidimDataset derives its sample size from base_dataset; "
                "construct a base dataset with the desired n instead."
            )

        base_dataset = self._make_base_dataset()
        covariates, treatments, outcomes = base_dataset.load()

        covariates = _as_polars_frame(covariates)
        treatments = _as_polars_frame(treatments)
        outcomes = _as_polars_frame(outcomes)

        if treatments.width != 1:
            raise ValueError(
                "MetaMultidimDataset requires a base dataset with exactly one "
                "treatment column."
            )

        continuous_column = treatments.columns[0]
        treatment_types = collect_column_types(treatments)
        if treatment_types[continuous_column] != "continuous":
            raise ValueError(
                "MetaMultidimDataset requires the base treatment column to be "
                "continuous."
            )

        categorical_column = self.categorical_column or f"{continuous_column}_bin"
        if categorical_column == continuous_column:
            raise ValueError(
                "categorical_column must differ from the continuous treatment "
                "column name."
            )

        self.base_dataset_ = base_dataset
        self.continuous_treatment_column_ = continuous_column
        self.categorical_treatment_column_ = categorical_column
        self.levels_ = [
            f"bin_{index}" for index in range(self.n_categorical_treatments)
        ]
        self.categorical_effects_ = self._sample_categorical_effects()

        categorical_series = self._make_categorical_treatment(
            treatments.get_column(continuous_column).to_numpy()
        )

        combined_treatments = pl.DataFrame(
            {
                continuous_column: treatments.get_column(continuous_column),
                categorical_column: categorical_series,
            }
        )

        self.column_types = {
            continuous_column: "continuous",
            categorical_column: "categorical",
        }
        self._covariates = covariates
        self._treatments = self._to_polars(combined_treatments)
        self._outcomes = self._scale_outcomes(outcomes, self._treatments)

        return self

    def _sample_categorical_effects(self) -> dict[str, float]:
        factors = self._rng.normal(
            loc=1.0,
            scale=self.categorical_effect_scale,
            size=self.n_categorical_treatments,
        )
        return {level: float(factor) for level, factor in zip(self.levels_, factors)}

    def _make_categorical_treatment(self, continuous_values) -> pl.Series:
        values = np.asarray(continuous_values, dtype=float).reshape(-1)
        quantiles = np.linspace(0.0, 1.0, self.n_categorical_treatments + 1)
        self.bin_edges_ = np.quantile(values, quantiles)
        interior_edges = self.bin_edges_[1:-1]

        bin_indices = np.searchsorted(interior_edges, values, side="right")
        labels = np.asarray(
            [self.levels_[index] for index in bin_indices], dtype=object
        )

        if self.mutual_info < 1.0:
            permute_mask = self._rng.random(values.shape[0]) > self.mutual_info
            n_permuted = int(np.count_nonzero(permute_mask))
            if n_permuted > 1:
                labels[permute_mask] = labels[permute_mask][
                    self._rng.permutation(n_permuted)
                ]

        return pl.Series(self.categorical_treatment_column_, labels)

    def _extract_continuous_treatments(self, treatments):
        if isinstance(treatments, pl.DataFrame):
            if self.continuous_treatment_column_ not in treatments.columns:
                raise ValueError(
                    "Treatments must include the wrapped continuous treatment "
                    f"column {self.continuous_treatment_column_!r}."
                )
            return treatments.select(self.continuous_treatment_column_)

        array = np.asarray(treatments, dtype=object)
        if array.ndim == 0:
            raise ValueError("Expected one treatment row per sample.")
        if array.ndim == 1:
            return array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("Expected treatments to be a 1D or 2D array-like object.")
        if array.shape[1] not in {1, len(self.column_types)}:
            raise ValueError(
                "Expected treatments with either the continuous column alone or "
                "the full mixed-treatment table."
            )
        return array[:, [0]]

    def _derive_categorical_labels_from_continuous(
        self, continuous_values
    ) -> list[str]:
        values = np.asarray(continuous_values, dtype=float).reshape(-1)
        indices = np.searchsorted(self.bin_edges_[1:-1], values, side="right")
        return [self.levels_[index] for index in indices]

    def _extract_categorical_labels(self, treatments) -> list[str]:
        if isinstance(treatments, pl.DataFrame):
            if self.categorical_treatment_column_ in treatments.columns:
                return (
                    treatments.get_column(self.categorical_treatment_column_)
                    .cast(pl.Utf8)
                    .to_list()
                )

            continuous_values = treatments.get_column(
                self.continuous_treatment_column_
            ).to_numpy()
            return self._derive_categorical_labels_from_continuous(continuous_values)

        array = np.asarray(treatments, dtype=object)
        if array.ndim == 0:
            raise ValueError("Expected one treatment row per sample.")
        if array.ndim == 1:
            return self._derive_categorical_labels_from_continuous(array)
        if array.ndim != 2:
            raise ValueError("Expected treatments to be a 1D or 2D array-like object.")
        if array.shape[1] == len(self.column_types):
            return [str(label) for label in array[:, 1].reshape(-1)]
        return self._derive_categorical_labels_from_continuous(array[:, 0])

    def _get_categorical_effects(self, treatments) -> np.ndarray:
        labels = self._extract_categorical_labels(treatments)
        return np.asarray(
            [self.categorical_effects_[label] for label in labels],
            dtype=float,
        )

    def _scale_outcomes(self, outcomes: pl.DataFrame, treatments) -> pl.DataFrame:
        scales = self._get_categorical_effects(treatments)
        return pl.DataFrame(
            {
                column: np.asarray(outcomes.get_column(column).to_numpy(), dtype=float)
                * scales
                for column in outcomes.columns
            }
        )

    def _predict_y(self, covariates, treatments) -> np.ndarray:
        continuous_treatments = self._extract_continuous_treatments(treatments)
        base_predictions = self.base_dataset_.predict_y(
            covariates, continuous_treatments
        )
        categorical_effects = self._get_categorical_effects(treatments)
        return np.asarray(base_predictions, dtype=float) * categorical_effects.reshape(
            -1, 1
        )

    def get_levels(self):
        return self._coerce_backend_frame(
            pl.DataFrame({self.categorical_treatment_column_: self.levels_}),
            column_types={self.categorical_treatment_column_: "categorical"},
        )

    def get_grid(self, n: int = 100):
        continuous_values = self._treatments.get_column(
            self.continuous_treatment_column_
        )
        continuous_grid = pl.DataFrame(
            {
                self.continuous_treatment_column_: np.linspace(
                    float(continuous_values.min()),
                    float(continuous_values.max()),
                    n,
                )
            }
        )
        categorical_levels = pl.DataFrame(
            {self.categorical_treatment_column_: self.levels_}
        )
        return self._coerce_treatment_frame(
            continuous_grid.join(categorical_levels, how="cross")
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        return [
            {
                "base_dataset": SyntheticDataset2(n=2000, random_state=7),
                "categorical_effect_scale": 0.2,
                "mutual_info": 0.5,
                "random_state": 7,
            },
            {
                "base_dataset": SyntheticDataset2(n=48, random_state=11),
                "categorical_effect_scale": 0.2,
                "n_categorical_treatments": 3,
                "mutual_info": 0.0,
                "random_state": 11,
            },
        ]
