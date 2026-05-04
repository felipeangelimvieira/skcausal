from skcausal.utils.lookup import (
    all_datasets,
    all_causal_average_response_estimators,
    all_density_estimators,
)


def test_all_density_estimators_returns_registered_density_classes():
    result = all_density_estimators(as_dataframe=True)

    assert not result.empty


def test_all_causal_average_response_estimators_returns_registered_classes():
    result = all_causal_average_response_estimators(as_dataframe=True)

    assert not result.empty
