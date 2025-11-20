from skcausal.datasets.synthetic2 import SyntheticDataset2, SyntheticDataset2Discrete
import polars as pl


def test_synthetic2_prepare_retrieve():
    dataset = SyntheticDataset2()
    dataset.prepare(n=1000)
    X, t, y = dataset.retrieve(test=False)
