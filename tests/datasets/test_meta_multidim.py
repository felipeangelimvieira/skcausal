import numpy as np
import polars as pl
import pytest

from skcausal.datasets.meta_multidim import MetaMultidimDataset
from skcausal.datasets.synthetic2 import SyntheticDataset2
from skcausal.datasets.synthetic2_multidim import Synthetic2MultidimDataset


def test_meta_multidim_requires_explicit_base_dataset():
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        MetaMultidimDataset()


def test_meta_multidim_load_and_predict_y_match_wrapped_dataset():
    dataset = MetaMultidimDataset(
        base_dataset=SyntheticDataset2(n=48, random_state=7),
        n_categorical_treatments=4,
        mutual_info=1.0,
        categorical_effect_scale=0.2,
        random_state=7,
    )
    covariates, treatments, outcomes = dataset.load()

    assert isinstance(covariates, pl.DataFrame)
    assert isinstance(treatments, pl.DataFrame)
    assert isinstance(outcomes, pl.DataFrame)
    assert treatments.columns == ["t_0", "t_0_bin"]
    assert treatments.schema["t_0"] == pl.Float64
    assert treatments.schema["t_0_bin"] == pl.Categorical

    labels = treatments.get_column("t_0_bin").cast(pl.Utf8).to_list()
    row_effects = np.asarray(
        [dataset.categorical_effects_[label] for label in labels],
        dtype=float,
    ).reshape(-1, 1)
    expected = dataset.base_dataset_.predict_y(covariates, treatments.select("t_0"))
    expected = expected * row_effects
    predictions_from_polars = dataset.predict_y(covariates, treatments)
    predictions_from_pandas = dataset.predict_y(
        covariates.to_pandas(), treatments.to_pandas()
    )
    predictions_from_numpy = dataset.predict_y(
        covariates.to_numpy(), treatments.to_numpy()
    )

    np.testing.assert_allclose(predictions_from_polars, expected)
    np.testing.assert_allclose(predictions_from_pandas, expected)
    np.testing.assert_allclose(predictions_from_numpy, expected)

    _, _, base_outcomes = dataset.base_dataset_.load()
    np.testing.assert_allclose(
        outcomes.to_numpy(), base_outcomes.to_numpy() * row_effects
    )

    grid = dataset.get_grid(1)
    assert isinstance(grid, pl.DataFrame)
    assert grid.shape == (4, 2)
    assert grid.schema["t_0_bin"] == pl.Categorical

    curve = dataset.predict_curve(covariates, grid)
    base_grid = pl.DataFrame({"t_0": [float(grid["t_0"][0])] * len(dataset.levels_)})
    base_curve = dataset.base_dataset_.predict_curve(covariates, base_grid)
    expected_curve = base_curve * np.asarray(
        [dataset.categorical_effects_[level] for level in dataset.levels_],
        dtype=float,
    )

    np.testing.assert_allclose(curve, expected_curve)
    assert curve.shape == (4,)


def test_meta_multidim_mutual_info_extremes_control_label_alignment():
    aligned_dataset = MetaMultidimDataset(
        base_dataset=SyntheticDataset2(n=200, random_state=13),
        n_categorical_treatments=5,
        mutual_info=1.0,
        random_state=13,
    )
    permuted_dataset = MetaMultidimDataset(
        base_dataset=SyntheticDataset2(n=200, random_state=13),
        n_categorical_treatments=5,
        mutual_info=0.0,
        random_state=13,
    )

    _, aligned_treatments, _ = aligned_dataset.load()
    _, permuted_treatments, _ = permuted_dataset.load()

    expected_indices = np.searchsorted(
        aligned_dataset.bin_edges_[1:-1],
        aligned_treatments.get_column("t_0").to_numpy(),
        side="right",
    )
    expected_labels = [f"bin_{index}" for index in expected_indices]
    aligned_labels = aligned_treatments.get_column("t_0_bin").cast(pl.Utf8).to_list()
    permuted_labels = permuted_treatments.get_column("t_0_bin").cast(pl.Utf8).to_list()

    assert aligned_labels == expected_labels
    assert permuted_labels != aligned_labels
    assert sorted(permuted_labels) == sorted(aligned_labels)


def test_synthetic2_multidim_wraps_synthetic2_by_default():
    dataset = Synthetic2MultidimDataset(
        n=48,
        n_categorical_treatments=4,
        mutual_info=1.0,
        categorical_effect_scale=0.2,
        random_state=7,
    )

    assert isinstance(dataset.base_dataset_, SyntheticDataset2)
    assert dataset.base_dataset_.n == 48

    _, treatments, _ = dataset.load()

    assert isinstance(treatments, pl.DataFrame)
    assert treatments.columns == ["t_0", "t_0_bin"]
