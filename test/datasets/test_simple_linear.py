import numpy as np
import polars as pl

from skcausal.datasets.simple_linear import SimpleLinearDataset


def test_simple_linear_load_returns_expected_dataframes_and_schema():
    dataset = SimpleLinearDataset(
        n=40,
        t_types=("continuous", "binary"),
        n_features=5,
        scale=0.2,
        random_state=17,
    )

    covariates, treatments, outcomes = dataset.load()

    assert isinstance(covariates, pl.DataFrame)
    assert isinstance(treatments, pl.DataFrame)
    assert isinstance(outcomes, pl.DataFrame)
    assert covariates.shape == (40, 5)
    assert treatments.shape == (40, 2)
    assert outcomes.shape == (40, 1)
    assert treatments.schema["t0"] == pl.Float64
    assert treatments.schema["t1"] == pl.Boolean


def test_simple_linear_predict_y_matches_exact_linear_formula_for_polars_and_numpy():
    dataset = SimpleLinearDataset(
        n=12,
        t_types=("continuous", "binary"),
        n_features=4,
        random_state=3,
    )
    covariates, treatments, _ = dataset.load()

    predictions_from_polars = dataset.predict_y(covariates, treatments)
    predictions_from_numpy = dataset.predict_y(
        covariates.to_numpy(), treatments.to_numpy()
    )
    expected = (
        covariates.to_numpy() @ dataset.beta_x_
        + treatments.to_numpy().astype(float) @ dataset.beta_t_
    )

    np.testing.assert_allclose(predictions_from_polars, expected)
    np.testing.assert_allclose(predictions_from_numpy, expected)
    assert predictions_from_polars.shape == (12,)


def test_simple_linear_binary_treatment_uses_threshold_rule():
    dataset = SimpleLinearDataset(
        n=25,
        t_types=("binary",),
        n_features=3,
        random_state=9,
    )
    covariates, treatments, _ = dataset.load()

    covariates_array = covariates.to_numpy()
    expected_binary_treatment = (
        covariates_array @ dataset.beta_x_ > covariates_array.mean()
    )

    np.testing.assert_array_equal(
        treatments.get_column("t0").to_numpy(),
        expected_binary_treatment,
    )


def test_simple_linear_exposes_requested_params():
    dataset = SimpleLinearDataset(
        t_types=("binary", "continuous"),
        n_features=7,
        scale=0.5,
        random_state=13,
    )

    params = dataset.get_params()

    assert params["t_types"] == ("binary", "continuous")
    assert params["n_features"] == 7
    assert params["scale"] == 0.5
    assert params["random_state"] == 13
