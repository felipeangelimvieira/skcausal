import numpy as np
import polars as pl
import pytest

from skcausal.datasets.ihdp import IHDPContinuous


def test_ihdp_continuous_load_uses_validated_hill_covariates():
    dataset = IHDPContinuous(seed=7)

    covariates, treatments, outcomes = dataset.load()

    assert isinstance(covariates, pl.DataFrame)
    assert isinstance(treatments, pl.DataFrame)
    assert isinstance(outcomes, pl.DataFrame)
    assert covariates.columns == [f"x{i}" for i in range(1, 26)]
    assert treatments.columns == ["t"]
    assert outcomes.columns == ["y"]
    assert covariates.shape == (747, 25)
    assert treatments.shape == (747, 1)
    assert outcomes.shape == (747, 1)
    assert treatments.schema["t"] == pl.Float64
    assert dataset.source_path_.name == "ihdp.csv"

    treatment_values = treatments.get_column("t").to_numpy()
    assert np.all(treatment_values >= 0.0)
    assert np.all(treatment_values <= 1.0)


def test_ihdp_continuous_predict_y_accepts_backends_and_curve_alias():
    dataset = IHDPContinuous(seed=11)
    covariates, treatments, _ = dataset.load()
    grid = dataset.get_grid(13)

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
    assert predictions_from_polars.shape == (747, 1)
    assert curve.shape == (13,)
    assert np.all(np.isfinite(curve))


def test_ihdp_continuous_retrieve_rejects_dataset_owned_test_split():
    dataset = IHDPContinuous(seed=3)

    with pytest.raises(NotImplementedError, match="external split"):
        dataset.retrieve(test=True)


def test_ihdp_continuous_rejects_malformed_raw_file(tmp_path):
    bad_path = tmp_path / "ihdp.csv"
    bad_path.write_text("1,2,3\n4,5,6\n", encoding="utf-8")

    with pytest.raises(ValueError, match="747 rows and 30 columns"):
        IHDPContinuous(source_path=bad_path)
