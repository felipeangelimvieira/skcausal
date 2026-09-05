from typing import ClassVar

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skcausal.datasets.semi_synthetic import BaseSemiSyntheticDataset
from tests.datasets._semi_synthetic_test_utils import (
    SecondToyRealDataset,
    ToyRealDataset,
)


class ToySemiSyntheticDataset(BaseSemiSyntheticDataset):
    column_types: ClassVar = {"t": "continuous"}

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
    column_types: ClassVar = {"t": "continuous"}

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
        column_types: ClassVar = {"t": "continuous"}

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


class SplittingSemiSyntheticDataset(ToySemiSyntheticDataset):
    def _split_source(self, X, treatment, outcome):
        self.seen_outcome_columns_ = list(outcome.columns)
        return X.drop("income")

    def _sample_treatment(self, X, rng):
        return pl.DataFrame({"t": X.get_column("age").to_numpy() / 50.0})

    def _log_prob(self, X, treatment):
        return np.zeros(X.height)


def test_base_split_source_hook_releases_reduced_covariates():
    dataset = SplittingSemiSyntheticDataset(ToyRealDataset(), random_state=1)
    X, treatment, _ = dataset.load()

    assert list(X.columns) == ["age", "region"]
    assert dataset.seen_outcome_columns_ == ["source_y"]
    assert dataset.n == 8
    assert treatment.shape == (8, 1)
    assert dataset.log_prob(X.to_numpy(), treatment).shape == (8, 1)


class ConcatenatingSemiSyntheticDataset(ToySemiSyntheticDataset):
    def _combine_sources(self, sources, rng):
        self.alignment_draw_ = rng.random()
        n_rows = min(X.height for X, _, _ in sources)
        return pl.concat(
            [
                X.head(n_rows).select(pl.all().name.prefix(f"s{index}_"))
                for index, (X, _, _) in enumerate(sources)
            ],
            how="horizontal",
        )

    def _prepare_dgp(self, X, rng):
        self.offset_ = float(X.get_column("s0_age").mean())

    def _sample_treatment(self, X, rng):
        return pl.DataFrame({"t": X.get_column("s0_income").to_numpy() / 50.0})

    def _log_prob(self, X, treatment):
        return np.zeros(X.height)

    def _predict_y(self, X, treatment):
        return treatment.get_column("t").to_numpy()


def test_base_rejects_invalid_source_sequences():
    with pytest.raises(TypeError, match="non-empty sequence"):
        ToySemiSyntheticDataset([])
    with pytest.raises(TypeError, match="non-empty sequence"):
        ToySemiSyntheticDataset([ToyRealDataset(), "not a dataset"])
    with pytest.raises(TypeError, match="non-empty sequence"):
        ToySemiSyntheticDataset(object())


def test_base_single_source_hooks_reject_multiple_sources():
    with pytest.raises(ValueError, match="exactly one source dataset; received 2"):
        ToySemiSyntheticDataset([ToyRealDataset(), ToyRealDataset()])
    single = ToySemiSyntheticDataset([ToyRealDataset()], random_state=1)
    assert len(single.real_datasets_) == 1
    assert single.real_dataset_ is single.real_datasets_[0]


def test_base_combine_sources_hook_receives_every_source_and_a_seeded_stream():
    sources = [ToyRealDataset(), SecondToyRealDataset()]
    dataset = ConcatenatingSemiSyntheticDataset(sources, random_state=3)
    X, treatment, _ = dataset.load()

    assert len(dataset.real_datasets_) == 2
    assert all(
        clone is not source for clone, source in zip(dataset.real_datasets_, sources)
    )
    assert dataset.real_dataset_ is dataset.real_datasets_[0]
    assert list(X.columns) == [
        "s0_age",
        "s0_income",
        "s0_region",
        "s1_height",
        "s1_score",
        "s1_group",
    ]
    assert dataset.n == 8 and treatment.shape == (8, 1)
    again = ConcatenatingSemiSyntheticDataset(sources, random_state=3)
    assert again.alignment_draw_ == dataset.alignment_draw_
    other = ConcatenatingSemiSyntheticDataset(sources, random_state=4)
    assert other.alignment_draw_ != dataset.alignment_draw_
