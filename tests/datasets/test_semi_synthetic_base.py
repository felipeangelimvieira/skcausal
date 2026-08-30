import numpy as np
import pandas as pd
import polars as pl
import pytest

from skcausal.datasets.semi_synthetic import BaseSemiSyntheticDataset
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


class ToySemiSyntheticDataset(BaseSemiSyntheticDataset):
    column_types = {"t": "continuous"}

    def _prepare_dgp(self, X, rng):
        self.offset_ = float(X.get_column("age").mean())

    def _sample_treatment(self, X, rng):
        values = X.get_column("income").to_numpy() / 50.0
        return pl.DataFrame({"t": values + rng.normal(scale=0.2, size=len(values))})

    def _log_prob(self, X, treatment):
        mean = X.get_column("income").to_numpy() / 50.0
        values = treatment.get_column("t").to_numpy()
        return -0.5 * ((values - mean) / 0.2) ** 2 - np.log(0.2 * np.sqrt(2 * np.pi))

    def _predict_y(self, X, treatment):
        age = X.get_column("age").to_numpy()
        values = treatment.get_column("t").to_numpy()
        return ((age - self.offset_) / 10.0 + values).reshape(-1, 1)

    def _sample_outcome(self, mean, X, treatment, rng):
        return np.asarray(mean) + rng.normal(scale=0.1, size=np.asarray(mean).shape)

    def _get_grid(self, n):
        return pl.DataFrame({"t": np.linspace(-1.0, 1.0, n)})


class PolarsOnlySemiSyntheticDataset(BaseSemiSyntheticDataset):
    column_types = {"t": "continuous"}

    def _prepare_dgp(self, X, rng):
        assert isinstance(X, pl.DataFrame)

    def _sample_treatment(self, X, rng):
        assert isinstance(X, pl.DataFrame)
        return pl.DataFrame({"t": np.zeros(X.height)})

    def _log_prob(self, X, treatment):
        assert isinstance(X, pl.DataFrame)
        assert isinstance(treatment, pl.DataFrame)
        return np.zeros(X.height)

    def _predict_y(self, X, treatment):
        assert isinstance(X, pl.DataFrame)
        assert isinstance(treatment, pl.DataFrame)
        return np.zeros((X.height, 1))

    def _sample_outcome(self, mean, X, treatment, rng):
        return np.asarray(mean)

    def _get_grid(self, n):
        return pl.DataFrame({"t": np.linspace(-1.0, 1.0, n)})


def test_base_preserves_source_covariates_and_oracle_shape():
    source = ToyRealDataset()
    dataset = ToySemiSyntheticDataset(source, random_state=7)
    X, treatment, outcome = dataset.load()

    assert X.equals(source.load()[0])
    assert dataset.real_dataset_ is not source
    assert treatment.columns == ["t"]
    assert outcome.columns == ["y"]
    assert dataset.log_prob(X, treatment).shape == (X.height, 1)
    assert dataset.predict_y(X, treatment).shape == (X.height, 1)
    assert dataset.get_grid(5).shape == (5, 1)


def test_sample_is_reproducible_and_does_not_mutate_load():
    dataset = ToySemiSyntheticDataset(ToyRealDataset(), random_state=11)
    loaded_before = dataset.load()
    first = dataset.sample(random_state=19)
    second = dataset.sample(random_state=19)
    loaded_after = dataset.load()

    for first_frame, second_frame in zip(first, second):
        assert first_frame.equals(second_frame)
    for before, after in zip(loaded_before, loaded_after):
        assert before.equals(after)
    assert not first[1].equals(loaded_before[1])


def test_base_rejects_prepare_sample_size_and_misaligned_oracle_rows():
    dataset = ToySemiSyntheticDataset(ToyRealDataset(), random_state=3)
    X, treatment, _ = dataset.load()

    with pytest.raises(ValueError, match="sample size"):
        dataset.prepare(n=10)
    with pytest.raises(ValueError, match="same number of rows"):
        dataset.log_prob(X.head(2), treatment.head(1))


def test_base_requires_a_dataset_object():
    with pytest.raises(TypeError, match="BaseDataset"):
        ToySemiSyntheticDataset(object(), random_state=3)


def test_base_requires_predict_y_hook_before_construction():
    class MissingPredictYDataset(BaseSemiSyntheticDataset):
        column_types = {"t": "continuous"}

        def _prepare_dgp(self, X, rng):
            pass

        def _sample_treatment(self, X, rng):
            return pl.DataFrame({"t": np.zeros(X.height)})

        def _log_prob(self, X, treatment):
            return np.zeros(X.height)

        def _sample_outcome(self, mean, X, treatment, rng):
            return np.asarray(mean)

        def _get_grid(self, n):
            return pl.DataFrame({"t": np.linspace(-1.0, 1.0, n)})

    with pytest.raises(TypeError, match="_predict_y"):
        MissingPredictYDataset(ToyRealDataset())


def test_oracle_hooks_receive_polars_for_numpy_and_pandas_inputs():
    dataset = PolarsOnlySemiSyntheticDataset(ToyRealDataset(), random_state=5)

    numpy_X = np.array([[1.0, 2.0], [3.0, 4.0]])
    numpy_treatment = np.array([0.1, 0.2])
    assert dataset.predict_y(numpy_X, numpy_treatment).shape == (2, 1)

    pandas_X = pd.DataFrame({"feature": [1.0, 2.0]})
    pandas_treatment = pd.DataFrame({"t": [0.1, 0.2]})
    assert dataset.log_prob(pandas_X, pandas_treatment).shape == (2, 1)
