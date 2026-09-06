from skcausal.datasets import (
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
    discovered = dict(all_datasets(return_names=True))

    assert "BaseSemiSyntheticDataset" not in discovered
    assert (
        discovered["CategoricalSemiSyntheticDataset"] is CategoricalSemiSyntheticDataset
    )
    assert (
        discovered["ContinuousSemiSyntheticDataset"] is ContinuousSemiSyntheticDataset
    )
    assert discovered["MultidimSemiSyntheticDataset"] is MultidimSemiSyntheticDataset
