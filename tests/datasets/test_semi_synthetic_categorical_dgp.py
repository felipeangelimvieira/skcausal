import numpy as np
import polars as pl
import pytest

from skcausal.datasets.semi_synthetic_categorical import (
    CategoricalSemiSyntheticDataset,
)
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


def _all_level_probabilities(dataset, X):
    columns = []
    for level in dataset.treatment_levels_:
        treatment = pl.DataFrame({"t": [level] * X.height})
        columns.append(np.exp(dataset.log_prob(X, treatment)[:, 0]))
    return np.column_stack(columns)


def test_categorical_dgp_preserves_X_and_has_exact_calibrated_probabilities():
    target = np.array([0.2, 0.3, 0.5])
    source = ToyRealDataset(include_missing=True)
    dataset = CategoricalSemiSyntheticDataset(
        source,
        n_treatments=3,
        target_probabilities=target,
        randomized_weight=0.15,
        random_state=7,
    )
    X, treatment, outcome = dataset.load()
    probabilities = _all_level_probabilities(dataset, X)

    assert X.equals(source.load()[0])
    assert treatment.schema["t"] == pl.Categorical
    assert (
        treatment.get_column("t").cast(pl.Utf8).is_in(dataset.treatment_levels_).all()
    )
    assert outcome.schema["y"] == pl.Float64
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(probabilities.mean(axis=0), target, atol=1e-9)


def test_categorical_randomized_component_is_independent_of_X():
    dataset = CategoricalSemiSyntheticDataset(
        ToyRealDataset(),
        n_treatments=3,
        target_probabilities=[0.25, 0.5, 0.25],
        randomized_weight=1.0,
        random_state=4,
    )
    X, _, _ = dataset.load()
    probabilities = _all_level_probabilities(dataset, X)

    np.testing.assert_allclose(
        probabilities,
        np.tile([0.25, 0.5, 0.25], (X.height, 1)),
        atol=1e-12,
    )


def test_categorical_outcome_oracle_accepts_all_backends():
    dataset = CategoricalSemiSyntheticDataset(
        ToyRealDataset(), outcome_noise_scale=0.0, random_state=12
    )
    X, treatment, outcome = dataset.load()

    polars_mean = dataset.predict_y(X, treatment)
    pandas_mean = dataset.predict_y(X.to_pandas(), treatment.to_pandas())
    numpy_mean = dataset.predict_y(X.to_numpy(), treatment.to_numpy())

    np.testing.assert_allclose(polars_mean, outcome.to_numpy())
    np.testing.assert_allclose(polars_mean, pandas_mean)
    np.testing.assert_allclose(polars_mean, numpy_mean)
    assert dataset.get_levels().shape == (2, 1)


def test_categorical_dgp_validates_configuration_and_unknown_levels():
    with pytest.raises(ValueError, match="at least 2"):
        CategoricalSemiSyntheticDataset(ToyRealDataset(), n_treatments=1)
    with pytest.raises(ValueError, match="sum to one"):
        CategoricalSemiSyntheticDataset(
            ToyRealDataset(), n_treatments=2, target_probabilities=[0.8, 0.8]
        )

    dataset = CategoricalSemiSyntheticDataset(ToyRealDataset(), random_state=2)
    X, _, _ = dataset.load()
    unknown = pl.DataFrame({"t": ["unsupported"] * X.height})
    assert np.isneginf(dataset.log_prob(X, unknown)).all()
