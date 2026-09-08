"""Contract tests for the supervised-table sources.

These cover the released ``(covariates, outcome)`` pair and the boundary
between that validation and whichever source supplied the table. Every test
here is offline.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

from skcausal import config
from skcausal.datasets import SupervisedFrame
from skcausal.datasets.base import BaseDataset, BaseTabularDataset
from skcausal.datasets.real.supervised import BaseSupervisedDataset
from skcausal.datasets.semisynthetic.model_induced import ModelInducedConfounding


def _frame(rows=12):
    index = np.arange(rows, dtype=float)
    return {
        "age": index + 20.0,
        "income": index * 3.0,
        "region": ["n", "s", "e"] * (rows // 3),
    }


def _numeric_dataset(rows=12):
    return SupervisedFrame(
        features=_frame(rows),
        target=np.linspace(-1.0, 1.0, rows),
        target_name="source_y",
    )


def test_load_releases_covariates_and_an_outcome_and_no_treatment():
    loaded = _numeric_dataset().load()

    # A supervised benchmark has no treatment, and none is invented for it.
    assert len(loaded) == 2
    covariates, outcomes = loaded
    assert covariates.columns == ["age", "income", "region"]
    assert outcomes.columns == ["source_y"]
    assert covariates.height == outcomes.height == 12
    # The semi-synthetic DGPs branch on categorical covariates to build their
    # encoder, so a label column must not collapse to float codes.
    assert covariates.get_column("region").dtype == pl.Utf8


def test_a_supervised_source_is_not_a_causal_dataset():
    dataset = _numeric_dataset()

    assert isinstance(dataset, BaseTabularDataset)
    assert not isinstance(dataset, BaseDataset)
    # The covariates-and-outcome accessor consumers use is the load pair here.
    covariates, outcomes = dataset.load_covariates_and_outcome()
    assert (covariates.columns, outcomes.columns) == (
        ["age", "income", "region"],
        ["source_y"],
    )


def test_load_respects_the_configured_default_backend(monkeypatch):
    monkeypatch.setattr(config, "default_backend", "pandas")

    covariates, outcomes = _numeric_dataset().load()

    assert isinstance(covariates, pd.DataFrame)
    assert isinstance(outcomes, pd.DataFrame)


def test_the_declared_target_kind_decides_the_outcome_dtype():
    categorical = SupervisedFrame(
        features=_frame(),
        target=["low", "high"] * 6,
        target_kind="categorical",
        target_name="label",
    ).load()[1]
    assert categorical.get_column("label").dtype == pl.Utf8
    assert set(categorical.get_column("label").to_list()) == {"low", "high"}

    # OpenML publishes ordinal scores such as wine quality as a nominal class.
    # An explicit numeric declaration should be honoured, not refused.
    numeric = SupervisedFrame(
        features=_frame(),
        target=["5", "6", "7"] * 4,
        target_kind="numeric",
        target_name="quality",
    ).load()[1]
    assert numeric.get_column("quality").dtype == pl.Float64
    assert numeric.get_column("quality").to_list()[:3] == [5.0, 6.0, 7.0]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"target": np.zeros(12)}, "at least two distinct values"),
        ({"target": np.linspace(0.0, 1.0, 5)}, "one value per feature row"),
        ({"target": np.full(12, np.nan)}, "must be finite"),
        ({"target": np.zeros((12, 2))}, "one-dimensional"),
        ({"target_kind": "ordinal"}, "target_kind must be one of"),
        ({"target": ["a", "b"] * 6, "target_kind": "numeric"}, "must hold numbers"),
        ({"target_name": "age"}, "must not also appear among the features"),
    ],
)
def test_malformed_source_is_rejected(kwargs, message):
    params = {
        "features": _frame(),
        "target": np.linspace(-1.0, 1.0, 12),
        "target_name": "source_y",
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        SupervisedFrame(**params).load()


def test_duplicate_feature_names_are_rejected():
    features = pd.DataFrame(np.arange(6.0).reshape(3, 2), columns=["a", "a"])

    with pytest.raises(ValueError, match="unique"):
        SupervisedFrame(features=features, target=[1.0, 2.0, 3.0]).load()


def test_the_target_is_named_from_every_input_shape():
    features = {"x": [1.0, 2.0, 3.0]}

    named = SupervisedFrame(
        features=features, target=pd.Series([1.0, 2.0, 3.0], name="score")
    ).load()[1]
    default = SupervisedFrame(features=features, target=[1.0, 2.0, 3.0]).load()[1]
    overridden = SupervisedFrame(
        features=features, target=np.array([1.0, 2.0, 3.0]), target_name="z"
    ).load()[1]

    assert named.columns == ["score"]
    assert default.columns == ["target"]
    assert overridden.columns == ["z"]

    # A falsy name such as the integer column 0 must survive, not fall back.
    frame = pd.DataFrame(np.random.default_rng(0).random((6, 3)))
    assert SupervisedFrame(features=frame[[1, 2]], target=frame[0]).load()[
        1
    ].columns == ["0"]


def test_a_second_source_reuses_the_mapping_unchanged():
    """The adapter must work for a source it has never heard of."""

    class ArraySource(BaseSupervisedDataset):
        def __init__(self, n):
            self.n = n
            super().__init__()

        def _fetch_supervised(self):
            values = np.arange(self.n, dtype=float)
            features = pl.DataFrame({"v": values, "w": values**2})
            return features, pl.Series("y", values / self.n), "numeric"

    covariates, outcomes = ArraySource(n=8).load()

    assert covariates.columns == ["v", "w"]
    assert outcomes.columns == ["y"]
    assert covariates.height == outcomes.height == 8


def test_feeds_a_semi_synthetic_dataset():
    """The whole point of these sources, exercised end to end with no network.

    The consumer fits a model on the covariate matrix directly, so the source
    here is all-numeric; a label column would have to be encoded first.
    """

    age = np.arange(60.0)
    source = SupervisedFrame(
        features={"age": age, "income": age * 3.0},
        target=np.where(age < 30.0, "low", "high"),
        target_name="source_y",
        target_kind="categorical",
    )
    dataset = ModelInducedConfounding(
        classifier=LogisticRegression(max_iter=2000),
        dataset=source,
        treatment_types=["continuous"],
        random_state=3,
    )
    covariates, treatments, outcomes = dataset.load()

    assert covariates.height == treatments.height == outcomes.height == 60
    # The DGP supplies the treatment the supervised source does not have.
    assert treatments.width == 1
