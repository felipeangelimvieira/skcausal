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


class SecondToyRealDataset(BaseDataset):
    """A ten-row source with its own covariate names and outcome."""

    def _load(self):
        X = pl.DataFrame(
            {
                "height": [
                    150.0,
                    155.0,
                    160.0,
                    165.0,
                    170.0,
                    175.0,
                    180.0,
                    185.0,
                    190.0,
                    195.0,
                ],
                "score": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0],
                "group": ["a", "b", "a", "c", "b", "c", "a", "b", "c", "a"],
            }
        )
        source_t = pl.DataFrame({"source_t": np.zeros(10)})
        source_y = pl.DataFrame({"other_y": np.linspace(2.0, -3.0, 10)})
        return X, source_t, source_y
