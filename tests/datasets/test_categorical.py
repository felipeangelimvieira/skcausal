import numpy as np
import polars as pl

from skcausal.datasets.categorical import ExampleCategorical


def test_confounded_categorical_dataset_load_returns_named_polars_frames():
    dataset = ExampleCategorical(n=40, random_state=13)

    covariates, treatments, outcomes = dataset.load()

    assert isinstance(covariates, pl.DataFrame)
    assert isinstance(treatments, pl.DataFrame)
    assert isinstance(outcomes, pl.DataFrame)
    assert covariates.columns == ["x0", "x1"]
    assert treatments.columns == ["treatment"]
    assert outcomes.columns == ["y"]
    assert treatments.schema["treatment"] == pl.Categorical


def test_confounded_categorical_dataset_predict_y_matches_closed_form():
    dataset = ExampleCategorical(n=24, random_state=5)
    covariates, treatments, _ = dataset.load()

    predictions_from_polars = dataset.predict_y(covariates, treatments)
    predictions_from_numpy = dataset.predict_y(
        covariates.to_numpy(), treatments.to_numpy()
    )
    expected = dataset.covariate_effect * covariates["x0"].to_numpy() + np.array(
        [
            dataset.treatment_effects[label]
            for label in treatments["treatment"].cast(pl.Utf8)
        ],
        dtype=float,
    )

    np.testing.assert_allclose(predictions_from_polars, expected)
    np.testing.assert_allclose(predictions_from_numpy, expected)


def test_confounded_categorical_dataset_levels_and_truth_are_ordered():
    dataset = ExampleCategorical(n=30, random_state=17)
    covariates, _, _ = dataset.load()

    levels = dataset.get_levels()
    truth = dataset.predict(covariates, levels)
    expected = np.array(
        [
            dataset.covariate_effect * float(covariates["x0"].mean())
            + dataset.treatment_effects[level]
            for level in ["control", "placebo", "treated"]
        ]
    )

    assert levels.schema["treatment"] == pl.Categorical
    assert levels["treatment"].cast(pl.Utf8).to_list() == [
        "control",
        "placebo",
        "treated",
    ]
    np.testing.assert_allclose(truth, expected)
