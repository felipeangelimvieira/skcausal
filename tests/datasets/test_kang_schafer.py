import numpy as np
import polars as pl

from skcausal.datasets.kang_schafer import (
    KangSchaferBinary,
    KangSchaferBinaryCrossValidation,
    KangSchaferBinaryMisspecified,
    KangSchaferContinuous,
    KangSchaferContinuousMisspecified,
)


def test_kang_schafer_binary_load_and_grid_use_boolean_treatment():
    dataset = KangSchaferBinary(n=48, random_state=7)

    covariates, treatments, outcomes = dataset.load()

    assert isinstance(covariates, pl.DataFrame)
    assert isinstance(treatments, pl.DataFrame)
    assert isinstance(outcomes, pl.DataFrame)
    assert covariates.columns == ["x1", "x2", "x3", "x4"]
    assert treatments.columns == ["a"]
    assert outcomes.columns == ["y"]
    assert treatments.schema["a"] == pl.Boolean
    assert set(dataset.get_grid().get_column("a").to_list()) == {False, True}


def test_kang_schafer_binary_predict_y_preserves_true_effect_one():
    dataset = KangSchaferBinary(n=24, random_state=11)
    covariates, _, _ = dataset.load()

    y1 = dataset.predict_y(covariates, pl.DataFrame({"a": [True] * 24}))
    y0 = dataset.predict_y(covariates, pl.DataFrame({"a": [False] * 24}))

    np.testing.assert_allclose(y1 - y0, 1.0)


def test_kang_schafer_binary_misspecified_exposes_transformed_covariates():
    dataset = KangSchaferBinaryMisspecified(n=32, random_state=3)
    covariates, treatments, _ = dataset.load()

    predictions_from_polars = dataset.predict_y(covariates, treatments)
    predictions_from_pandas = dataset.predict_y(
        covariates.to_pandas(), treatments.to_pandas()
    )
    predictions_from_numpy = dataset.predict_y(
        covariates.to_numpy(), treatments.to_numpy()
    )

    assert covariates.columns == ["z1", "z2", "z3", "z4"]
    np.testing.assert_allclose(predictions_from_polars, predictions_from_pandas)
    np.testing.assert_allclose(predictions_from_polars, predictions_from_numpy)


def test_kang_schafer_continuous_predict_y_accepts_multiple_backends():
    dataset = KangSchaferContinuous(n=40, random_state=5)
    covariates, treatments, _ = dataset.load()

    predictions_from_polars = dataset.predict_y(covariates, treatments)
    predictions_from_pandas = dataset.predict_y(
        covariates.to_pandas(), treatments.to_pandas()
    )
    predictions_from_numpy = dataset.predict_y(
        covariates.to_numpy(), treatments.to_numpy()
    )

    np.testing.assert_allclose(predictions_from_polars, predictions_from_pandas)
    np.testing.assert_allclose(predictions_from_polars, predictions_from_numpy)
    assert treatments.schema["a"] == pl.Float64


def test_kang_schafer_continuous_misspecified_predict_curve_matches_alias():
    dataset = KangSchaferContinuousMisspecified(n=36, random_state=9)
    covariates, _, _ = dataset.load()
    grid = dataset.get_grid(9)

    curve = dataset.predict_curve(covariates, grid)
    legacy_curve = dataset.predict(covariates, grid)

    assert covariates.columns == ["z1", "z2", "z3", "z4"]
    np.testing.assert_allclose(curve, legacy_curve)
    assert curve.shape == (9,)


def test_kang_schafer_cross_validation_dataset_has_fixed_size():
    dataset = KangSchaferBinaryCrossValidation(random_state=13)
    covariates, treatments, outcomes = dataset.load()

    assert dataset.n == 2000
    assert covariates.shape == (2000, 4)
    assert treatments.shape == (2000, 1)
    assert outcomes.shape == (2000, 1)
