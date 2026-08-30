import numpy as np
import polars as pl

from skcausal.datasets.base import BaseDataset


class ToyRealDataset(BaseDataset):
    def __init__(self, include_missing=False):
        self.include_missing = include_missing
        super().__init__()

    def _load(self):
        age = [21.0, 25.0, 29.0, 33.0, 37.0, 41.0, 45.0, 49.0]
        if self.include_missing:
            age[2] = None
        X = pl.DataFrame(
            {
                "age": age,
                "income": [32.0, 38.0, 44.0, 50.0, 56.0, 62.0, 68.0, 74.0],
                "region": ["n", "s", "n", "e", "s", "e", "n", "s"],
            }
        )
        source_t = pl.DataFrame({"source_t": np.arange(8, dtype=float)})
        source_y = pl.DataFrame({"source_y": np.linspace(-1.0, 1.0, 8)})
        return X, source_t, source_y
