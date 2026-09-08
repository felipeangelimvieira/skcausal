import weakref
from typing import ClassVar

import numpy as np
import polars as pl
import pytest

from skcausal.datasets.base import BaseKnownResponseDataset


@pytest.mark.parametrize("backend", ["numpy", "polars"])
def test_predict_curve_releases_expanded_rows(backend):
    expanded_rows = []

    class Dataset(BaseKnownResponseDataset):
        column_types: ClassVar = {"dose": "continuous", "group": "categorical"}

        def predict_y(self, covariates, treatments):
            expanded_rows.append(weakref.ref(treatments))
            assert sum(ref() is not None for ref in expanded_rows) == 1
            assert treatments.shape == (len(covariates), 2)
            if isinstance(treatments, pl.DataFrame):
                assert treatments.schema == grid.schema
                treatments = treatments.to_numpy()
            return treatments[:, :1].astype(float)

    grid = pl.DataFrame({"dose": [1.0, 2.0, 3.0], "group": ["a", "b", "a"]})
    grid = grid.with_columns(pl.col("group").cast(pl.Categorical))
    if backend == "numpy":
        grid = grid.to_numpy()

    curve = Dataset().predict_curve(np.zeros((8, 1)), grid)

    np.testing.assert_array_equal(curve, [1.0, 2.0, 3.0])
