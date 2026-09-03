from skcausal.datasets import (
    BaseSemiSyntheticDataset,
    CategoricalSemiSyntheticDataset,
    ContinuousSemiSyntheticDataset,
    MultidimSemiSyntheticDataset,
)
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
    discovered = all_datasets(return_names=True)
    names = {name for name, _ in discovered}

    assert BaseSemiSyntheticDataset.__name__ == "BaseSemiSyntheticDataset"
    assert CategoricalSemiSyntheticDataset.__name__ == "CategoricalSemiSyntheticDataset"
    assert ContinuousSemiSyntheticDataset.__name__ == "ContinuousSemiSyntheticDataset"
    assert MultidimSemiSyntheticDataset.__name__ == "MultidimSemiSyntheticDataset"
    assert "CategoricalSemiSyntheticDataset" in names
    assert "ContinuousSemiSyntheticDataset" in names
    assert "MultidimSemiSyntheticDataset" in names
