import numpy as np
import polars as pl
import pytest

from skcausal.utils.treatment_grid import (
    make_cartesian_treatment_grid,
    sample_treatment_rows,
)


def test_make_cartesian_treatment_grid_builds_mixed_column_grid():
    t = pl.DataFrame(
        {
            "dose": [0.0, 0.5, 1.0],
            "arm": ["control", "treated", "control"],
        }
    ).with_columns(pl.col("arm").cast(pl.Categorical))

    grid = make_cartesian_treatment_grid(t, n_continuous_points=3)

    assert grid.height == 6
    np.testing.assert_allclose(grid["dose"].unique().sort().to_numpy(), [0.0, 0.5, 1.0])
    assert grid["arm"].n_unique() == 2
    assert dict(grid.schema) == {"dose": pl.Float64, "arm": pl.Categorical}


def test_sample_treatment_rows_draws_unique_rows_without_replacement():
    t = pl.DataFrame(
        {
            "dose": [0.0, 0.0, 0.5, 1.0],
            "arm": ["control", "control", "treated", "treated"],
        }
    )

    sample = sample_treatment_rows(
        t,
        n_rows=2,
        random_state=7,
    )

    assert sample.height == 2
    assert sample.unique().height == sample.height

    candidate_rows = {tuple(row) for row in t.unique(maintain_order=True).iter_rows()}
    sampled_rows = {tuple(row) for row in sample.iter_rows()}
    assert sampled_rows.issubset(candidate_rows)


def test_make_cartesian_treatment_grid_uses_registered_types_for_integer_columns():
    t = pl.DataFrame({"dose": [0, 2, 6]})

    grid = make_cartesian_treatment_grid(
        t,
        n_continuous_points=4,
    )

    np.testing.assert_allclose(grid["dose"].to_numpy(), [0.0, 2.0, 4.0, 6.0])
    assert dict(grid.schema) == {"dose": pl.Float64}


def test_sample_treatment_rows_requires_positive_n_rows():
    t = pl.DataFrame({"dose": [0.0, 1.0]})

    with pytest.raises(ValueError, match="n_rows"):
        sample_treatment_rows(t, n_rows=0)
