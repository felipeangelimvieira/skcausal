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


class _HookedSemiSyntheticRegressor(SemiSyntheticRegressor):
    def _load_dataset(self):
        return _toy_regression_dataset()


def test_semisynthetic_regressor_normalizes_dataset_and_uses_predictions_for_treatment():
    dataset = _HookedSemiSyntheticRegressor(
        regressor=LinearRegression(),
        random_state=7,
        treatment_effect_scale=0.0,
    )

    covariates, treatments, outcomes = dataset.load()

    assert isinstance(covariates, pl.DataFrame)
    assert isinstance(treatments, pl.DataFrame)
    assert isinstance(outcomes, pl.DataFrame)
    assert covariates.columns == ["age", "income", "score"]
    assert treatments.columns == ["t"]
    assert outcomes.columns == ["y"]

    covariate_array = covariates.to_numpy()
    outcome_array = outcomes.get_column("y").to_numpy()
    treatment_array = treatments.get_column("t").to_numpy()

    np.testing.assert_allclose(covariate_array.mean(axis=0), np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(
        covariate_array.std(axis=0, ddof=0), np.ones(3), atol=1e-10
    )
    np.testing.assert_allclose(outcome_array.mean(), 0.0, atol=1e-10)
    np.testing.assert_allclose(outcome_array.std(ddof=0), 1.0, atol=1e-10)

    expected_treatments = dataset.regressor_.predict(covariate_array)
    np.testing.assert_allclose(treatment_array, expected_treatments)


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
