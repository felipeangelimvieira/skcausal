import numpy as np
import polars as pl

from skcausal.datasets.polynomial import PolynomialDataset


def test_polynomial_dataset_load_returns_dataframes():
    dataset = PolynomialDataset(n=32, seed=7)

    covariates, treatments, outcomes = dataset.load()

    assert isinstance(covariates, pl.DataFrame)
    assert isinstance(treatments, pl.DataFrame)
    assert isinstance(outcomes, pl.DataFrame)
    assert covariates.shape == (32, 3)
    assert treatments.shape == (32, 3)
    assert outcomes.shape == (32, 1)
    assert treatments.schema["t2"] == pl.Boolean


def test_polynomial_dataset_predict_y_accepts_polars_and_numpy():
    dataset = PolynomialDataset(n=16, seed=11)
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
    assert predictions_from_polars.shape == (16, 1)
