from skcausal.datasets import ModelInducedConfounding
from skcausal.datasets.base import BaseDataset
from skcausal.utils.lookup import (
    all_causal_average_response_estimators,
    all_datasets,
    all_density_estimators,
)


def test_all_density_estimators_returns_registered_density_classes():
    result = all_density_estimators(as_dataframe=True)

    assert not result.empty


def test_all_causal_average_response_estimators_returns_registered_classes():
    result = all_causal_average_response_estimators(as_dataframe=True)

    assert not result.empty


def test_semi_synthetic_datasets_are_public_and_discoverable():
    discovered = dict(all_datasets(return_names=True))

    assert "BaseSemiSyntheticDataset" not in discovered
    assert discovered["ModelInducedConfounding"] is ModelInducedConfounding


def test_all_datasets_returns_only_causal_datasets():
    """A dataset in this registry has a treatment; a supervised table does not.

    Estimators reach for `all_datasets` to find problems they can be run on, so
    a source that releases no treatment must not appear here.
    """

    discovered = dict(all_datasets(return_names=True))

    assert discovered
    assert all(issubclass(cls, BaseDataset) for cls in discovered.values())
    assert "SupervisedFrame" not in discovered
    assert "OpenMLDataset" not in discovered
    assert "BaseTabularDataset" not in discovered
