import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

from skcausal.datasets.real.sklearn import SklearnDataset
from skcausal.datasets.semisynthetic.model_induced import (
    ModelInducedConfounding,
)

N_WINE_ROWS = 178
THREE = ["continuous", "categorical", "continuous"]
THREE_CORRELATION = [[1.0, 0.6, 0.3], [0.6, 1.0, 0.2], [0.3, 0.2, 1.0]]


def _wine_source():
    return SklearnDataset(name="wine")


def _dataset(**overrides):
    # A regularized classifier is not incidental. Wine is very nearly
    # separable, and an unregularized fit puts most rows above 0.99 on their
    # predicted class. That is a degenerate propensity: the sampled
    # categorical coordinate becomes a deterministic function of the
    # covariates, and the coupling has no randomness left to act on.
    params = {
        "classifier": LogisticRegression(max_iter=2000, C=0.01),
        "dataset": _wine_source(),
        "random_state": 7,
        "outcome_noise_scale": 0.0,
    }
    params.update(overrides)
    return ModelInducedConfounding(**params)


def _reference_score(dataset, covariates, class_index=0):
    """Recompute the documented score: standardized log-odds of one class."""

    probabilities = dataset.classifier_.predict_proba(covariates)[:, class_index]
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(probabilities) - np.log1p(-probabilities)
    return (logits - logits.mean()) / logits.std(ddof=0)


def test_treatments_are_the_declared_mixed_pair():
    dataset = _dataset()
    covariates, treatments, outcomes = dataset.load()

    assert treatments.columns == ["t_0", "t_1"]
    assert treatments.schema["t_0"] == pl.Float64
    assert treatments.schema["t_1"] == pl.Categorical
    assert outcomes.columns == ["y"]
    assert covariates.columns == _wine_source().load()[0].columns
    assert dataset.treatment_levels_ == ["0", "1", "2"]


def test_zero_noise_makes_the_continuous_coordinate_the_fitted_score():
    dataset = _dataset(treatment_noise_scale=0.0)
    covariates, treatments, _ = dataset.load()

    np.testing.assert_allclose(
        treatments.get_column("t_0").to_numpy(),
        _reference_score(dataset, covariates.to_numpy()),
        atol=1e-10,
    )


def test_categorical_coordinate_is_drawn_from_the_fitted_propensity():
    dataset = _dataset()
    covariates, treatments, _ = dataset.load()

    probabilities = dataset.classifier_.predict_proba(covariates.to_numpy())
    labels = treatments.get_column("t_1").cast(pl.Utf8).to_numpy()

    for index, level in enumerate(dataset.treatment_levels_):
        assert np.mean(labels == level) == pytest.approx(
            probabilities[:, index].mean(), abs=0.06
        )


def test_the_propensity_does_not_move_with_the_coupling():
    independent = _dataset(treatment_correlation=0.0)
    coupled = _dataset(treatment_correlation=0.9)

    for dataset in (independent, coupled):
        _, treatments, _ = dataset.load()
        labels = treatments.get_column("t_1").cast(pl.Utf8).to_numpy()
        probabilities = dataset.classifier_.predict_proba(dataset.load()[0].to_numpy())
        for index, level in enumerate(dataset.treatment_levels_):
            assert np.mean(labels == level) == pytest.approx(
                probabilities[:, index].mean(), abs=0.06
            )


def test_treatment_correlation_raises_the_achieved_mutual_information():
    weak = _dataset(treatment_correlation=0.0)
    strong = _dataset(treatment_correlation=0.9)

    assert strong.achieved_mutual_information_.shape == (2, 2)
    assert (
        strong.achieved_mutual_information_[0, 1]
        > weak.achieved_mutual_information_[0, 1]
    )


def test_treatment_correlation_is_inert_without_assignment_noise():
    kwargs = {"treatment_noise_scale": 0.0}

    _, without, _ = _dataset(treatment_correlation=0.0, **kwargs).load()
    _, with_coupling, _ = _dataset(treatment_correlation=0.9, **kwargs).load()

    assert without.get_column("t_0").equals(with_coupling.get_column("t_0"))


def test_predict_y_is_additive_in_the_two_coordinates():
    dataset = _dataset(treatment_effect_scale=1.5)
    covariates, treatments, _ = dataset.load()

    first, second = dataset.treatment_levels_[:2]
    at_first = dataset.predict_y(
        covariates, treatments.with_columns(pl.lit(first).alias("t_1"))
    )
    at_second = dataset.predict_y(
        covariates, treatments.with_columns(pl.lit(second).alias("t_1"))
    )

    difference = at_first - at_second
    assert not np.allclose(difference, 0.0)
    np.testing.assert_allclose(
        difference, np.full_like(difference, difference[0, 0]), atol=1e-10
    )


def test_predict_y_accepts_every_backend_and_the_grid_crosses_both_coordinates():
    dataset = _dataset()
    covariates, treatments, _ = dataset.load()
    grid = dataset.get_grid(9)
    levels = dataset.get_levels()

    from_polars = dataset.predict_y(covariates, treatments)

    np.testing.assert_allclose(
        from_polars, dataset.predict_y(covariates.to_pandas(), treatments.to_pandas())
    )
    np.testing.assert_allclose(
        from_polars, dataset.predict_y(covariates.to_numpy(), treatments.to_numpy())
    )
    assert from_polars.shape == (N_WINE_ROWS, 1)
    assert grid.columns == ["t_0", "t_1"]
    assert grid.height == 9 * len(dataset.treatment_levels_)
    assert grid.schema["t_0"] == pl.Float64 and grid.schema["t_1"] == pl.Categorical
    assert levels.get_column("t_1").cast(pl.Utf8).to_list() == dataset.treatment_levels_
    assert dataset.predict_curve(covariates, grid).shape == (grid.height,)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"treatment_correlation": 1.0}, "treatment_correlation"),
        ({"treatment_correlation": -1.5}, "treatment_correlation"),
        ({"treatment_noise_scale": -0.1}, "treatment_noise_scale"),
        ({"treatment_types": []}, "treatment_types"),
        ({"treatment_types": ["continuous", "ordinal"]}, "treatment_types"),
        (
            {
                "treatment_correlation": [
                    [1.0, 0.5, 0.0],
                    [0.5, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            },
            "treatment_correlation",
        ),
        (
            {"treatment_correlation": [[1.0, 0.5], [0.4, 1.0]]},
            "treatment_correlation",
        ),
        (
            {
                "treatment_types": THREE,
                "treatment_correlation": [
                    [1.0, 0.9, -0.9],
                    [0.9, 1.0, 0.9],
                    [-0.9, 0.9, 1.0],
                ],
            },
            "positive definite",
        ),
    ],
)
def test_invalid_coupling_parameters_are_rejected(params, match):
    with pytest.raises(ValueError, match=match):
        _dataset(**params)


def test_three_coordinates_release_the_declared_columns():
    dataset = _dataset(treatment_types=THREE, treatment_correlation=THREE_CORRELATION)
    _, treatments, _ = dataset.load()

    assert treatments.columns == ["t_0", "t_1", "t_2"]
    assert treatments.schema["t_0"] == pl.Float64
    assert treatments.schema["t_1"] == pl.Categorical
    assert treatments.schema["t_2"] == pl.Float64
    assert dataset.column_types == dict(zip(["t_0", "t_1", "t_2"], THREE))
    assert dataset.achieved_mutual_information_.shape == (3, 3)
    assert dataset.get_levels().columns == ["t_1"]


def test_later_continuous_coordinates_use_the_next_class_log_odds():
    dataset = _dataset(treatment_types=THREE, treatment_noise_scale=0.0)
    covariates, treatments, _ = dataset.load()

    np.testing.assert_allclose(
        treatments.get_column("t_2").to_numpy(),
        _reference_score(dataset, covariates.to_numpy(), class_index=1),
        atol=1e-10,
    )


def test_propensity_is_exact_under_a_dense_correlation_matrix():
    dataset = _dataset(treatment_types=THREE, treatment_correlation=THREE_CORRELATION)
    covariates, treatments, _ = dataset.load()

    probabilities = dataset.classifier_.predict_proba(covariates.to_numpy())
    labels = treatments.get_column("t_1").cast(pl.Utf8).to_numpy()
    for index, level in enumerate(dataset.treatment_levels_):
        assert np.mean(labels == level) == pytest.approx(
            probabilities[:, index].mean(), abs=0.06
        )


@pytest.mark.parametrize("rho", [0.9, -0.9])
def test_a_pairwise_entry_raises_that_pair_s_mutual_information(rho):
    # A large assignment noise lets the copula dominate the covariate channel.
    # At the default noise the two log-odds are anti-correlated through the
    # covariates, and a positive coupling would cancel rather than add.
    kwargs = {"treatment_types": THREE, "treatment_noise_scale": 3.0}
    weak = _dataset(treatment_correlation=0.0, **kwargs)
    strong = _dataset(
        treatment_correlation=[[1.0, 0.0, rho], [0.0, 1.0, 0.0], [rho, 0.0, 1.0]],
        **kwargs,
    )

    assert (
        strong.achieved_mutual_information_[0, 2]
        > weak.achieved_mutual_information_[0, 2]
    )


def test_predict_y_is_additive_in_every_coordinate():
    dataset = _dataset(treatment_types=THREE, treatment_effect_scale=1.5)
    covariates, treatments, _ = dataset.load()

    low = dataset.predict_y(
        covariates, treatments.with_columns(pl.lit(-1.0).alias("t_2"))
    )
    high = dataset.predict_y(
        covariates, treatments.with_columns(pl.lit(1.0).alias("t_2"))
    )

    difference = low - high
    assert not np.allclose(difference, 0.0)
    np.testing.assert_allclose(
        difference, np.full_like(difference, difference[0, 0]), atol=1e-10
    )


def test_grid_crosses_every_coordinate():
    dataset = _dataset(treatment_types=THREE)
    grid = dataset.get_grid(5)

    assert grid.columns == ["t_0", "t_1", "t_2"]
    assert grid.height == 5 * 5 * len(dataset.treatment_levels_)
    assert dataset.predict_curve(dataset.load()[0], grid).shape == (grid.height,)


@pytest.mark.parametrize("kind", ["continuous", "categorical"])
def test_one_coordinate_is_allowed(kind):
    """The single-coordinate case: one treatment column, nothing to couple."""

    dataset = _dataset(treatment_types=[kind], treatment_effect_scale=1.5)
    covariates, treatments, _ = dataset.load()

    assert treatments.columns == ["t_0"]
    assert treatments.schema["t_0"] == (
        pl.Float64 if kind == "continuous" else pl.Categorical
    )
    assert dataset.achieved_mutual_information_.shape == (1, 1)
    if kind == "continuous":
        np.testing.assert_allclose(
            _dataset(treatment_types=[kind], treatment_noise_scale=0.0)
            .load()[1]
            .get_column("t_0")
            .to_numpy(),
            _reference_score(dataset, covariates.to_numpy()),
            atol=1e-10,
        )

    grid = dataset.get_grid(6)
    assert grid.height == (
        6 if kind == "continuous" else len(dataset.treatment_levels_)
    )
    assert dataset.predict_curve(covariates, grid).shape == (grid.height,)
