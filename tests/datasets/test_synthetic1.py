import numpy as np
import polars as pl
import pytest

from skcausal.datasets.synthetic_vcnet import SyntheticVCNet


def test_synthetic_vcnet_load_exposes_single_dataset_sample():
    dataset = SyntheticVCNet(random_state=7)

    covariates, treatments, outcomes = dataset.load()

    assert isinstance(covariates, pl.DataFrame)
    assert isinstance(treatments, pl.DataFrame)
    assert isinstance(outcomes, pl.DataFrame)
    assert covariates.columns == [f"x{i}" for i in range(1, 7)]
    assert treatments.columns == ["t"]
    assert outcomes.columns == ["y"]
    assert covariates.shape == (500, 6)
    assert treatments.shape == (500, 1)
    assert outcomes.shape == (500, 1)
    assert treatments.schema["t"] == pl.Float64

    treatment_values = treatments.get_column("t").to_numpy()
    assert np.all(treatment_values >= 0.0)
    assert np.all(treatment_values <= 1.0)


def test_synthetic_vcnet_predict_y_accepts_backends_and_curve_alias():
    dataset = SyntheticVCNet(n=64, random_state=11)
    covariates, treatments, _ = dataset.load()
    grid = dataset.get_grid(9)

    predictions_from_polars = dataset.predict_y(covariates, treatments)
    predictions_from_pandas = dataset.predict_y(
        covariates.to_pandas(), treatments.to_pandas()
    )
    predictions_from_numpy = dataset.predict_y(
        covariates.to_numpy(), treatments.to_numpy()
    )
    curve = dataset.predict_curve(covariates, grid)
    legacy_curve = dataset.predict(covariates, grid)

    np.testing.assert_allclose(predictions_from_polars, predictions_from_pandas)
    np.testing.assert_allclose(predictions_from_polars, predictions_from_numpy)
    np.testing.assert_allclose(curve, legacy_curve)
    assert predictions_from_polars.shape == (64, 1)
    assert curve.shape == (9,)
    assert np.all(np.isfinite(curve))


def test_synthetic_vcnet_load_rejects_dataset_owned_test_split_kwarg():
    dataset = SyntheticVCNet(n=32, random_state=5)

    with pytest.raises(TypeError, match="unexpected keyword argument 'test'"):
        dataset.load(test=True)
