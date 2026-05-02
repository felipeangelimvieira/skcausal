import numpy as np
import polars as pl
import pytest

from skcausal.datasets.nurse_staffing import NurseStaffing


def test_nurse_staffing_load_returns_named_polars_frames():
    dataset = NurseStaffing(n=64, seed=7)

    covariates, treatments, outcomes = dataset.load()

    assert isinstance(covariates, pl.DataFrame)
    assert isinstance(treatments, pl.DataFrame)
    assert isinstance(outcomes, pl.DataFrame)
    assert covariates.columns == ["l1", "l2", "l3", "l4"]
    assert treatments.columns == ["a"]
    assert outcomes.columns == ["y"]
    assert covariates.shape == (64, 4)
    assert treatments.shape == (64, 1)
    assert outcomes.shape == (64, 1)
    assert treatments.schema["a"] == pl.Float64
    assert set(outcomes.get_column("y").unique().to_list()).issubset({0, 1})

    treatment_values = treatments.get_column("a").to_numpy()
    assert np.all(treatment_values >= 0.0)
    assert np.all(treatment_values <= dataset.TREATMENT_MAX)


def test_nurse_staffing_predict_y_accepts_polars_pandas_and_numpy():
    dataset = NurseStaffing(n=32, seed=11)
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
    assert np.all(predictions_from_polars >= 0.0)
    assert np.all(predictions_from_polars <= 1.0)


def test_nurse_staffing_outcome_type_switches_observed_scale():
    probability_dataset = NurseStaffing(
        n=24,
        seed=5,
        outcome_type="probability",
    )
    covariates, treatments, outcomes = probability_dataset.load()

    np.testing.assert_allclose(
        outcomes.to_numpy(),
        probability_dataset.predict_y(covariates, treatments),
    )

    logit_dataset = NurseStaffing(
        n=24,
        seed=5,
        outcome_type="logit",
    )
    covariates, treatments, outcomes = logit_dataset.load()
    logits = logit_dataset.predict_y(covariates, treatments)

    np.testing.assert_allclose(outcomes.to_numpy(), logits)
    assert np.any(logits < 0.0) or np.any(logits > 1.0)


def test_nurse_staffing_predict_curve_matches_predict_alias_on_grid():
    dataset = NurseStaffing(n=40, seed=3)
    covariates, _, _ = dataset.load()
    grid = dataset.get_grid(11)

    curve = dataset.predict_curve(covariates, grid)
    legacy_curve = dataset.predict(covariates, grid)

    np.testing.assert_allclose(curve, legacy_curve)
    assert curve.shape == (11,)
    assert grid.schema["a"] == pl.Float64
    assert grid.get_column("a")[0] == pytest.approx(0.0)
    assert grid.get_column("a")[-1] == pytest.approx(dataset.TREATMENT_MAX)


def test_nurse_staffing_rejects_unknown_outcome_type():
    with pytest.raises(ValueError, match="outcome_type"):
        NurseStaffing(outcome_type="score")
