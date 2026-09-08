"""Tests for the scikit-learn bundled-dataset source.

Every dataset exercised here ships inside scikit-learn, so these run offline.
``california_housing`` is the one supported name that downloads, and is only
checked for registration, never loaded.
"""

import polars as pl
import pytest

from skcausal.datasets import SklearnDataset


def test_a_classification_dataset_loads_with_a_categorical_target():
    covariates, outcome = SklearnDataset(name="wine").load_covariates_and_outcome()

    assert covariates.shape == (178, 13)
    assert covariates.columns[0] == "alcohol"
    assert outcome.width == 1
    assert outcome.schema[outcome.columns[0]] == pl.Utf8
    assert sorted(outcome.get_column(outcome.columns[0]).unique()) == ["0", "1", "2"]


def test_a_regression_dataset_loads_with_a_numeric_target():
    covariates, outcome = SklearnDataset(name="diabetes").load_covariates_and_outcome()

    assert covariates.shape == (442, 10)
    assert outcome.schema[outcome.columns[0]] == pl.Float64


def test_an_unknown_name_is_rejected_with_the_supported_ones():
    with pytest.raises(ValueError, match="wine"):
        SklearnDataset(name="not_a_dataset")


def test_the_dataset_name_is_required_and_shown_in_the_repr():
    # A default would be omitted from the repr, so the docs catalog would print
    # SklearnDataset() and never say which data the example used.
    with pytest.raises(TypeError):
        SklearnDataset()
    assert repr(SklearnDataset(name="wine")) == "SklearnDataset(name='wine')"


def test_california_housing_is_offered_but_not_loaded_on_construction():
    # It is the one supported name that downloads, so construction must stay
    # free of I/O and the name must still be accepted.
    assert "california_housing" in SklearnDataset.available_names()
    SklearnDataset(name="california_housing")
