import numpy as np
import polars as pl

from skcausal.datasets.synthetic2 import SyntheticDataset2, SyntheticDataset2Discrete


def test_synthetic2_load_and_predict_y_accept_polars_outputs():
    dataset = SyntheticDataset2(n=32)
    covariates, treatments, _ = dataset.load()

    predictions_from_polars = dataset.predict_y(covariates, treatments)
    predictions_from_pandas = dataset.predict_y(
        covariates.to_pandas(), treatments.to_pandas()
    )
    predictions_from_numpy = dataset.predict_y(
        covariates.to_numpy(), treatments.to_numpy()
    )

    np.testing.assert_allclose(predictions_from_polars, predictions_from_numpy)
    np.testing.assert_allclose(predictions_from_polars, predictions_from_pandas)
    assert predictions_from_polars.shape == (32, 1)
    assert dataset.get_grid(10).schema["t_0"] == pl.Float64


def test_synthetic2_predict_curve_matches_predict_for_grid():
    dataset = SyntheticDataset2(n=32, random_state=3)
    covariates, _, _ = dataset.load()
    grid = dataset.get_grid(15)

    curve = dataset.predict_curve(covariates, grid)
    pandas_curve = dataset.predict_curve(covariates.to_pandas(), grid.to_pandas())
    legacy_curve = dataset.predict(covariates, grid)

    np.testing.assert_allclose(curve, legacy_curve)
    np.testing.assert_allclose(curve, pandas_curve)
    assert curve.shape == (15,)


def test_synthetic2_discrete_load_and_grid_use_boolean_schema():
    dataset = SyntheticDataset2Discrete(n=24)
    covariates, treatments, outcomes = dataset.load()

    assert isinstance(covariates, pl.DataFrame)
    assert isinstance(treatments, pl.DataFrame)
    assert isinstance(outcomes, pl.DataFrame)
    assert treatments.shape == (24, 1)
    assert treatments.schema["treatment"] == pl.Boolean

    predictions_from_polars = dataset.predict_y(covariates, treatments)
    predictions_from_pandas = dataset.predict_y(
        covariates.to_pandas(), treatments.to_pandas()
    )
    predictions_from_numpy = dataset.predict_y(
        covariates.to_numpy(), treatments.to_numpy()
    )
    np.testing.assert_allclose(predictions_from_polars, predictions_from_numpy)
    np.testing.assert_allclose(predictions_from_polars, predictions_from_pandas)
    assert predictions_from_polars.shape == (24, 1)

    grid = dataset.get_grid()
    assert isinstance(grid, pl.DataFrame)
    assert grid.schema["treatment"] == pl.Boolean
    assert set(grid.get_column("treatment").to_list()) == {True, False}


def test_synthetic2_discrete_exposes_random_state_param():
    dataset = SyntheticDataset2Discrete(random_state=13)

    params = dataset.get_params()

    assert params["random_state"] == 13
