"""Contract tests for the shared supervised-source base class.

These cover what BaseSemiSyntheticDataset owns, through its subclass, so the
behaviour is pinned once rather than per test module.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

from skcausal.datasets import ModelInducedConfounding, SupervisedFrame
from skcausal.datasets.real.supervised import BaseSupervisedDataset
from skcausal.datasets.synthetic.synthetic2 import SyntheticDataset2


def _regression_source(rows=40):
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "age": rng.normal(40.0, 5.0, rows),
            "income": rng.normal(50.0, 8.0, rows),
        }
    )
    target = 0.4 * features["age"] + 0.2 * features["income"]
    return SupervisedFrame(features=features, target=target.to_numpy())


def _classification_source(rows=48):
    rng = np.random.default_rng(1)
    score = rng.normal(size=rows)
    features = pd.DataFrame({"score": score, "noise": rng.normal(size=rows)})
    labels = np.where(score < -0.4, "low", np.where(score < 0.4, "mid", "high"))
    return SupervisedFrame(features=features, target=labels, target_kind="categorical")


def _build(**overrides):
    params = {
        "classifier": LogisticRegression(max_iter=2000),
        "dataset": _classification_source(),
    }
    params.update(overrides)
    return ModelInducedConfounding(**params)


def test_source_must_be_a_dataset_object():
    # A source is a dataset object, not a factory for one. A callable must
    # fail loudly at construction rather than be mistaken for a source.
    with pytest.raises(TypeError, match="must be a BaseSupervisedDataset instance"):
        _build(dataset=lambda: (np.zeros((4, 2)), np.arange(4.0)))


def test_source_must_be_supervised_rather_than_causal():
    # A causal dataset is a tabular dataset too, but it already carries a
    # treatment. Taking one as a source silently discards that treatment and
    # replaces it with a model-induced one, which is not a source relationship
    # anybody asks for on purpose.
    with pytest.raises(TypeError, match="must be a BaseSupervisedDataset instance"):
        _build(dataset=SyntheticDataset2(n=20))


def test_source_task_tag_must_match_the_expected_task():
    with pytest.raises(ValueError, match="needs a classification source"):
        _build(dataset=_regression_source())


def test_a_source_that_cannot_name_its_task_is_accepted():
    # OpenML's target_kind="auto" tags None: unknown is not a mismatch.
    source = _classification_source()
    source.set_tags(task=None)

    assert _build(dataset=source).load()[0].height == 48


def test_covariate_names_come_from_the_source():
    dataset = _build()
    covariates, _, _ = dataset.load()

    assert covariates.columns == dataset.dataset.load()[0].columns


class _WideOutcomeSource(BaseSupervisedDataset):
    """A supervised source that breaks its own base's one-target contract.

    BaseSupervisedDataset builds the outcome from a single series, so only a
    subclass overriding _load can release a wide one. That is the case the
    guard downstream exists for, and the only way left to reach it.
    """

    def _load(self):
        rows = 8
        return (
            pl.DataFrame({"x": np.arange(rows, dtype=float)}),
            pl.DataFrame(
                {"a": np.arange(rows, dtype=float), "b": np.arange(rows, dtype=float)}
            ),
        )


def test_multi_column_source_outcome_is_rejected():
    with pytest.raises(ValueError, match="exactly one outcome column"):
        _build(dataset=_WideOutcomeSource())


def test_source_is_cloned_before_loading():
    # Preparation must not mutate the caller's source object.
    source = _build().dataset
    before = source.get_params()

    _build(dataset=source)

    assert source.get_params() == before


def test_the_dataset_derives_treatments_from_the_source():
    dataset = _build(treatment_effect_scale=1.0, random_state=3)
    covariates, treatments, outcomes = dataset.load()

    assert covariates.height == treatments.height == outcomes.height == 48
    assert np.isfinite(treatments.get_column("t_0").to_numpy()).all()
    assert np.isfinite(outcomes.to_numpy()).all()


def test_treatment_levels_come_from_the_source_labels():
    dataset = _build(treatment_effect_scale=0.0, outcome_noise_scale=0.0)
    _, treatments, _ = dataset.load()

    observed = set(treatments.get_column("t_1").cast(pl.Utf8).to_list())
    assert observed <= {"low", "mid", "high"}
    assert observed


def test_one_source_object_feeds_repeated_consumers():
    """A source object is not consumed or mutated by the dataset built on it."""

    source = _classification_source()

    first = _build(dataset=source)
    second = _build(dataset=source)

    assert first.load()[0].height == second.load()[0].height == 48
    assert first.load()[0].equals(second.load()[0])
