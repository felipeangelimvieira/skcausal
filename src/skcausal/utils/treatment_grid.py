"""Utilities for constructing feasible treatment evaluation tables."""

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
from skcausal.datatypes.utils import collect_column_types

__all__ = [
    "make_cartesian_treatment_grid",
    "sample_treatment_rows",
]


def make_cartesian_treatment_grid(
    t: pl.DataFrame,
    *,
    n_continuous_points: int | Mapping[str, int] = 25,
) -> pl.DataFrame:
    """Build a treatment evaluation table via a Cartesian product.

    Continuous treatment columns are replaced by evenly spaced points spanning
    their observed min/max range. Categorical treatment columns contribute their
    unique observed levels. The returned table is the Cartesian product of those
    per-column values.

    Parameters
    ----------
    t : pl.DataFrame
        Observed treatment table.
    n_continuous_points : int or mapping, default=25
        Number of grid points to generate per continuous column. A mapping can
        provide per-column values.

    Returns
    -------
    pl.DataFrame
        Treatment table suitable for evaluating ADRF curves on a smaller,
        structured grid.
    """

    if t.height == 0:
        raise ValueError("Cannot build a treatment grid from an empty treatment table.")

    continuous, categorical = _resolve_treatment_column_roles(t)
    point_spec = _normalize_continuous_points(continuous, n_continuous_points)

    column_frames = []
    for column in t.columns:
        if column in continuous:
            column_frames.append(
                _make_uniform_continuous_column_frame(
                    t,
                    column,
                    n_points=point_spec[column],
                )
            )
        else:
            column_frames.append(_make_categorical_column_frame(t, column))

    return _cross_join_frames(column_frames)


def sample_treatment_rows(
    t: pl.DataFrame,
    *,
    n_rows: int,
    unique_rows: bool = True,
    random_state: int | None = 0,
) -> pl.DataFrame:
    """Sample observed treatment rows without replacement.

    Parameters
    ----------
    t : pl.DataFrame
        Observed treatment table.
    n_rows : int
        Number of rows to sample.
    unique_rows : bool, default=True
        Whether to deduplicate identical treatment rows before sampling.
    random_state : int, optional
        Seed passed to Polars sampling for reproducibility.

    Returns
    -------
    pl.DataFrame
        A sampled subset of treatment rows.
    """

    if t.height == 0:
        raise ValueError("Cannot sample treatment rows from an empty treatment table.")
    if n_rows <= 0:
        raise ValueError("n_rows must be a positive integer.")

    candidates = t.unique(maintain_order=True) if unique_rows else t
    if n_rows >= candidates.height:
        return candidates

    return candidates.sample(
        n=n_rows,
        with_replacement=False,
        shuffle=True,
        seed=random_state,
    )


def _resolve_treatment_column_roles(t: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Resolve continuous and categorical treatment columns.

    All columns are classified using the project's registered datatype system.

    Parameters
    ----------
    t : pl.DataFrame
        Treatment table whose columns will be classified.

    Returns
    -------
    tuple of list of str
        Two lists containing the resolved continuous columns first and the
        resolved categorical columns second.
    """

    inferred_types = collect_column_types(t)
    inferred_continuous = []
    inferred_categorical = []

    for column in t.columns:
        column_type = inferred_types[column]
        if column_type == "continuous":
            inferred_continuous.append(column)
        elif column_type == "categorical":
            inferred_categorical.append(column)
        else:
            raise TypeError(
                f"Unsupported treatment column type {column_type!r} for column {column!r}."
            )

    return inferred_continuous, inferred_categorical


def _normalize_continuous_points(
    continuous_columns: Sequence[str],
    n_continuous_points: int | Mapping[str, int],
) -> dict[str, int]:
    """Normalize continuous grid counts into a per-column mapping.

    Parameters
    ----------
    continuous_columns : sequence of str
        Continuous treatment columns that need grid sizes.
    n_continuous_points : int or mapping of str to int
        Either one shared grid size for all continuous columns or an explicit
        per-column mapping.

    Returns
    -------
    dict of str to int
        Mapping from continuous column name to validated positive grid size.
    """

    if isinstance(n_continuous_points, Mapping):
        point_spec = {}
        for column in continuous_columns:
            if column not in n_continuous_points:
                raise ValueError(
                    "n_continuous_points mapping must include every continuous column. "
                    f"Missing {column!r}."
                )
            point_spec[column] = _validate_positive_integer(
                n_continuous_points[column],
                f"n_continuous_points[{column!r}]",
            )
        return point_spec

    n_points = _validate_positive_integer(
        n_continuous_points,
        "n_continuous_points",
    )
    return {column: n_points for column in continuous_columns}


def _validate_positive_integer(value, argument_name: str) -> int:
    """Return ``value`` when it is a positive integer, else raise.

    Parameters
    ----------
    value : object
        Candidate value to validate.
    argument_name : str
        Name of the calling argument, used in validation messages.

    Returns
    -------
    int
        The validated positive integer.
    """

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{argument_name} must be a positive integer.")
    return value


def _make_uniform_continuous_column_frame(
    t: pl.DataFrame,
    column: str,
    *,
    n_points: int,
) -> pl.DataFrame:
    """Create a one-column frame of evenly spaced values for one treatment column.

    Parameters
    ----------
    t : pl.DataFrame
        Treatment table providing the observed min and max values.
    column : str
        Continuous treatment column to expand into a uniform grid.
    n_points : int
        Number of grid points to generate between the observed endpoints.

    Returns
    -------
    pl.DataFrame
        One-column frame containing the generated grid values.
    """

    dtype = t.schema[column]
    values = t.get_column(column)
    min_value = values.min()
    max_value = values.max()

    if min_value is None or max_value is None:
        raise ValueError(f"Treatment column {column!r} does not contain any values.")

    if float(min_value) == float(max_value):
        return pl.DataFrame({column: pl.Series(column, [min_value], dtype=dtype)})

    grid_values = np.linspace(float(min_value), float(max_value), n_points)
    series = pl.Series(column, grid_values)
    if dtype == pl.Float32 or dtype == pl.Float64:
        series = series.cast(dtype)
    return pl.DataFrame({column: series})


def _make_categorical_column_frame(t: pl.DataFrame, column: str) -> pl.DataFrame:
    """Return the observed unique levels for one categorical treatment column.

    Parameters
    ----------
    t : pl.DataFrame
        Treatment table containing the categorical column.
    column : str
        Categorical treatment column whose observed levels should be extracted.

    Returns
    -------
    pl.DataFrame
        One-column frame containing the unique observed levels.
    """

    values = t.select(column).unique(maintain_order=True)
    if values.height == 0:
        raise ValueError(f"Treatment column {column!r} does not contain any values.")
    return values


def _cross_join_frames(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Cartesian-product a sequence of one-or-more column frames.

    Parameters
    ----------
    frames : sequence of pl.DataFrame
        One-column or multi-column frames whose rows should be combined via a
        Cartesian product.

    Returns
    -------
    pl.DataFrame
        The full Cartesian product of the provided frames.
    """

    if not frames:
        raise ValueError("At least one treatment column is required to build a grid.")

    product = frames[0]
    for frame in frames[1:]:
        product = product.join(frame, how="cross")
    return product
