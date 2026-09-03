import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression

from skcausal.datasets.semi_synthetic_regressor import SemiSyntheticRegressor


def _toy_regression_dataset():
    covariates = pd.DataFrame(
        {
            "age": [21.0, 25.0, 29.0, 33.0, 37.0, 41.0],
            "income": [35.0, 41.0, 47.0, 53.0, 59.0, 65.0],
            "score": [1.0, 0.5, 1.5, 1.0, 2.0, 1.5],
        }
    )
    outcome = (
        0.4 * covariates["age"] + 0.2 * covariates["income"] - 1.3 * covariates["score"]
    )
    outcome = outcome.to_numpy(dtype=float)
    return covariates, outcome


def test_semisynthetic_regressor_predict_y_accepts_backends_and_adds_spline_effect():
    dataset = SemiSyntheticRegressor(
        regressor=LinearRegression(),
        load_dataset=_toy_regression_dataset,
        random_state=3,
        treatment_effect_scale=1.25,
    )

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

    baseline = dataset.regressor_.predict(covariates.to_numpy()).reshape(-1, 1)

    np.testing.assert_allclose(predictions_from_polars, predictions_from_pandas)
    np.testing.assert_allclose(predictions_from_polars, predictions_from_numpy)
    assert predictions_from_polars.shape == (6, 1)
    assert curve.shape == (9,)
    assert not np.allclose(predictions_from_polars, baseline)
    assert grid.schema["t"] == pl.Float64
